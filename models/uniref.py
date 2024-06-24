import torch
from dataset.utils import get_uni_attetion_mask
from models.xvlm import load_pretrained
from typing import Optional, Tuple
from models import XVLMBase, build_mlp

from transformers import BertPreTrainedModel
from transformers.file_utils import ModelOutput


class UniRef(XVLMBase):
    def __init__(self, config, load_vision_params=True, load_text_params=True):
        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=True, use_bbox_loss=True, config_text=None)

        # self.region_proj = build_mlp(self.vision_width, self.text_width)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.region_threshold = self.text_encoder.bert.config.region_threshold
        
        self.text_encoder.bert.encoder.set_vit_pos_embed(self.vision_encoder.pos_embed.weight)
        print('Set Bert Encoder position embed!!!')

    def forward(self, image, text_ids=None, text_atts=None, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, 
                ret_mlm_loss=True, ret_bbox_loss=False, task_id=0, use_new_segment=0):

        image_embeds, image_atts, image_embeds_fullatts = \
            self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        
        ret = ()

        if ret_mlm_loss:
            token_type_ids = torch.zeros_like(text_ids_masked)
            if task_id==0:  # mlm
                lm_mask = text_atts
            else:  # reg  
                lm_mask = get_uni_attetion_mask(text_atts)
                if use_new_segment:
                    token_type_ids += 1
            visual_embeds = [image_embeds_fullatts, image_embeds]
            visual_atts = [torch.ones(image_embeds.shape[:2]).to(image_embeds.device), image_atts]
            loss_mlm = self.get_mlm_loss(text_ids_masked, lm_mask, visual_embeds, visual_atts, masked_pos, masked_ids, token_type_ids=token_type_ids)
            ret += (loss_mlm,)

        if ret_bbox_loss:
            text_embeds = self.get_text_embeds(text_ids, text_atts) 
            output_coord, region_logits = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts, ret_region_logits=True)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)
            ret += (loss_bbox, loss_giou)
 
            loss_pred, acc, recall = 0, 0, 0
            target = image_atts[:, 1:].float()
            for x in region_logits:
                loss_pred += self.bce_loss(x, target)
                pred = (torch.sigmoid(x) >= self.region_threshold).float()
                acc += (pred * target).sum(dim=1) / (pred.sum(dim=1) + 1e-9)
                recall += (pred * target).sum(dim=1) / target.sum(dim=1)
            loss_pred /= len(region_logits)
            acc /= len(region_logits)
            recall /= len(region_logits)
            f1 = 2 * acc * recall / (acc + recall + 1e-9)

            ret += (loss_pred, acc.mean(), recall.mean(), f1.mean())


        return ret

    def predict(self, image, text_ids, text_atts, ret_region_att=False, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            image_embeds, _ = self.get_vision_embeds(image)
        if text_embeds is None:
            text_embeds = self.get_text_embeds(text_ids, text_atts)

        output_coord, region_logits = self.predict_bbox(image_embeds, text_embeds, text_atts, ret_region_logits=True)
        if ret_region_att:
            region_att = (torch.stack(list(region_logits)).sigmoid() >= self.region_threshold).float()
            region_att = torch.transpose(region_att, 0, 1) # (bs, 6, patch_num)
            return (output_coord, region_att)

        return output_coord

    def load_pretrained(self, ckpt_rpath, config, is_eval):
        self.text_encoder.bert.encoder.set_vit_pos_embed(None)
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

        self.text_encoder.bert.encoder.set_vit_pos_embed(self.vision_encoder.pos_embed.weight)
        print('Set Bert Encoder position embed!!!')


class EncoderOutputs(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    image_atts: Optional[torch.FloatTensor] = None 

class XVLMOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class UniRefDecoder(BertPreTrainedModel):

    def __init__(self, config, xvlm_config, tokenizer, xvlm):
        super().__init__(config)

        self.xvlm = xvlm
        self.config = config
        self.mask_token_id = tokenizer.mask_token_id

    @classmethod
    def from_pretrained(cls, xvlm_config, tokenizer, xvlm):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(xvlm_config['text_encoder'])
        config.is_encoder_decoder = True
        model = cls(config, xvlm_config, tokenizer, xvlm)
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        is_decoder=True,
        **kwargs
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        assert return_dict, 'return_dict must be True'
        assert is_decoder, 'is_decoder must be True'

        encoder_outputs = kwargs.get('encoder_outputs')
        image_embeds = encoder_outputs.get('image_embeds')
        image_atts = encoder_outputs.get('image_atts')

        outputs = self.xvlm.text_encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        prediction_scores = self.xvlm.text_encoder.cls(sequence_output)

        return XVLMOutputs(
            loss=None,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1) + 1
        device = input_ids.device
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        attention_mask = causal_mask[:,-2:,:]

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        new_input_ids = input_ids.clone()  # (B2, 1)
        mask_ids = torch.zeros_like(new_input_ids) + self.mask_token_id
        new_input_ids = torch.cat([new_input_ids, mask_ids], dim=1)  # (B2, 2)

        # past_key_values
        new_past_key_values = None
        if past:
            new_past_key_values = ()
            # for (k, q, v, o) in past_key_values:
            #     nk, nq, nv, no = k[:,:,:-1,:], q[:,:,:-1,:], v[:,:,:-1,:], o[:,:,:-1,:]
            #     new_past_key_values += ((nk, nq, nv, no),)
            for (k, q) in past:
                nk, nq = k[:,:,:-1,:], q[:,:,:-1,:]
                new_past_key_values += ((nk, nq),)

        # image_atts = model_kwargs.get('image_atts')
        # image_embeds = model_kwargs.get('image_embeds')
        encoder_outputs = model_kwargs.get('encoder_outputs')
        segment_id = 1 if model_kwargs.get('use_new_segment') else 0
        token_type_ids = torch.zeros_like(new_input_ids) + segment_id

        return {
            "input_ids": new_input_ids, 
            "attention_mask": attention_mask, 
            "token_type_ids": token_type_ids,
            "past_key_values": new_past_key_values,
            'encoder_outputs': encoder_outputs
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ):
        if 'encoder_outputs' not in model_kwargs:
            image = model_kwargs.get('image')
            image_atts = model_kwargs.get('image_atts')
            idx_to_group_img = model_kwargs.get('idx_to_group_img')
            image_embeds, image_atts, image_embeds_fullatts = \
                self.xvlm.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
      
            model_kwargs['image_embeds'] = [image_embeds_fullatts, image_embeds]
            model_kwargs['image_atts'] = [torch.ones_like(image_atts), image_atts]
            encoder_outputs = EncoderOutputs(
                image_embeds=[image_embeds_fullatts, image_embeds],
                image_atts=[torch.ones_like(image_atts), image_atts],
                last_hidden_state=image_embeds_fullatts
            )
            model_kwargs['encoder_outputs'] = encoder_outputs
        return model_kwargs
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            encoder_outputs['image_embeds'][0] = encoder_outputs['image_embeds'][0].index_select(0, expanded_return_idx)
            encoder_outputs['image_embeds'][1] = encoder_outputs['image_embeds'][1].index_select(0, expanded_return_idx)
            encoder_outputs['image_atts'][0] = encoder_outputs['image_atts'][0].index_select(0, expanded_return_idx)
            encoder_outputs['image_atts'][1] = encoder_outputs['image_atts'][1].index_select(0, expanded_return_idx)

            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
