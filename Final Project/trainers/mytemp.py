import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_trainer = "MYTEMP"
    design_details = {"trainer": design_trainer,
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MYTEMP.MAPLE_LENGTH}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model, len_prompts):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.len_prompts = len_prompts

    def forward(self, text_x, compound_prompts_deeper_text, rpo_text_ctx_prompt, text_mask):
        device = "cuda"
        text_x = text_x.to(device)
        
        K = 4 ##
        for i in range(K):
            text_x[torch.arange(text_x.shape[0]), self.len_prompts+i, :] = rpo_text_ctx_prompt[i, :].repeat(text_x.shape[0], 1)

        text_x = text_x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [text_x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined, text_mask)
        text_x = outputs[0]  # extract the x back from here
        text_x = text_x.permute(1, 0, 2)  # LND -> NLD
        text_x = self.ln_final(text_x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        ##x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        text_f = torch.empty(text_x.shape[0], 0, 512, device=device, dtype=self.dtype)
        for i in range(4 * 2):
            idx = self.len_prompts + i
            x = text_x[torch.arange(text_x.shape[0]), idx]
            text_f = torch.cat([text_f, x[:, None, :]], dim=1)

        text_f = text_f @ self.text_projection

        return text_f

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MYTEMP.PROMPT_DEPTH >= 1, "For MYTEMP, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MYTEMP.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        ##RPO
        self.rpo_text_ctx_prompt, self.rpo_visual_ctx_prompt = self.rpo_init(cfg, clip_model)

        # random initialization
        maple_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(maple_ctx_vectors, std=0.02)

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.maple_ctx = nn.Parameter(maple_ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        # classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # # These token vectors will be saved when in save_model(),
        # # but they should be ignored in load_model() as we want to use
        # # those computed using the current class names
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        #self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

    def rpo_init(self, cfg, clip_model):
        positional_embedding = clip_model.positional_embedding

        ## Make sure K is even
        self.K = 4 # the number of prompt pair
        self.dtype = clip_model.dtype
        self.d_t = clip_model.ln_final.weight.shape[0] #512
        self.d_v = 768

        clip_imsize = clip_model.visual.input_resolution # 224
        cfg_imsize = cfg.INPUT.SIZE[0] # (224, 224)[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        return self.initialization_token(clip_model)

    def forward(self):
        #maple_ctx = self.maple_ctx

        # if maple_ctx.dim() == 2:
        #     maple_ctx = maple_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        #prefix = self.token_prefix
        #suffix = self.token_suffix

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        ## None indicates prompt location
        return None, self.proj(self.maple_ctx), self.compound_prompts_text, visual_deep_prompts, self.rpo_text_ctx_prompt, self.rpo_visual_ctx_prompt   # pass here original, as for visual 768 is required
        
    def initialization_token(self, clip_model):
        #### text token initialization #####
        
        # text_token = clip_model.token_embedding(torch.tensor([49407]))
        # text_token = text_token.repeat(self.K, 1)
        # text_noise = torch.randn(self.K, self.d_t)
        # text_noise = text_noise / text_noise.norm(dim=-1, keepdim=True)
        # text_token += 0.1 * text_noise
        # text_token = text_token.type(self.dtype)
        # text_prompt = nn.Parameter(text_token)

        t_prompt_vec = torch.empty(self.K, self.d_t, dtype=self.dtype)
        nn.init.normal_(t_prompt_vec, std=0.02)
        text_prompt = nn.Parameter(t_prompt_vec, requires_grad=True)

        #### visual token initialization ####
        
        # visual_token = clip_model.visual.class_embedding
        # visual_token = visual_token.repeat(self.K, 1)
        # visual_noise = torch.randn(self.K, self.d_v)
        # visual_noise = visual_noise / visual_noise.norm(dim=-1, keepdim=True)
        # visual_token += 0.1 * visual_noise
        # visual_token = visual_token.type(self.dtype)
        # img_prompt = nn.Parameter(visual_token)

        v_prompt_vec = torch.empty(self.K, self.d_v, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        img_prompt = nn.Parameter(v_prompt_vec, requires_grad=True)

        return text_prompt, img_prompt


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, prompt, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        ##RPO

        self.token_embedding = clip_model.token_embedding
        self.text_pos_embedding = clip_model.positional_embedding        
        self.cfg = cfg
        self.text_x = self.make_prompts(classnames, prompt) # ["a photo of a dog.", ".."]
        self.text_proj = clip_model.text_projection
        self.define_mask()
        ##MAPLE
        self.text_encoder = TextEncoder(clip_model, self.len_prompts)
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        #self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        

    def forward(self, image, label=None):

        ##RPO
        device = image.device
        #tokenized_prompts = self.tokenized_prompts


        _, shared_maple_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, rpo_text_ctx_prompt, rpo_visual_ctx_prompt = self.prompt_learner()
        
        text_f = self.text_encoder(self.text_x, deep_compound_prompts_text, rpo_text_ctx_prompt, self.text_mask)
        img_f = self.image_encoder(image.type(self.dtype), shared_maple_ctx, deep_compound_prompts_vision, rpo_visual_ctx_prompt, self.visual_mask)


        ####################### logit ###########################
        # logit
        K = self.cfg.TRAINER.MYTEMP.K
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        #print(text_f.shape, img_f.shape)
        logits = torch.zeros(img_f.shape[0], text_f.shape[0], device=device)
        for i in range(K):
            i_img_f = img_f[:,i,:]
            i_text_f = text_f[:,i,:]
            logit = self.logit_scale.exp() * i_img_f @ i_text_f.t()
            logits += logit
        logits /= K

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits

    def make_prompts(self, classnames, prompt):
        prompts = [prompt.replace('_', c) for c in classnames]
        with torch.no_grad():
            self.text_tokenized = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = self.token_embedding(self.text_tokenized).type(self.dtype) + self.text_pos_embedding.type(self.dtype)
            self.len_prompts = self.text_tokenized.argmax(dim=-1) + 1
        return prompts

    def define_mask(self):
        len_max = 77
        attn_head = 8

        text_mask = torch.empty(0, len_max, len_max)
        for idx in self.len_prompts:
            mask = torch.empty(len_max, len_max)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            mask[:, idx:].fill_(float("-inf"))
            text_mask = torch.cat([text_mask, mask.repeat(attn_head, 1, 1)])
        self.text_mask = text_mask

        # image encoder mask
        att_size = 1 + 14 * 14 + self.cfg.TRAINER.MYTEMP.K
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)
        visual_mask[:, -1 * self.cfg.TRAINER.MYTEMP.K:] = float("-inf")
        #####

        self.visual_mask = visual_mask


@TRAINER_REGISTRY.register()
class MYTEMP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MYTEMP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MYTEMP.PREC == "fp32" or cfg.TRAINER.MYTEMP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        prompt = cfg.DATASET.PROMPT
        self.model = CustomCLIP(cfg, classnames, prompt, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MYTEMP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MYTEMP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

  
