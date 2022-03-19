import copy
import os
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel

from CEPC.src.ELib import ELib


class EClassifier(nn.Module):

    def __init__(self):
        super(EClassifier, self).__init__()
        self.logs = dict()
        self.train_state = 0
        self.epoch_index = -1
        self.train_step = -1
        self.hooked_modules = dict()
        self.hook_interval = None
        self.hook_activated = None
        self.hooksForward = list()
        self.hooksBackward = list()
        ELib.PASS()

    def setup_logs(self, dir_path, curve_names, add_hooks=False, hook_interval=10):
        # one summary_writer will be created for each name in curve_names
        # also if add_hooks=True one summary_writer will be also created
        # for each module in self.hooked_modules
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        for cur_name in curve_names:
            self.logs[cur_name] = SummaryWriter(os.path.join(dir_path, cur_name))
        if add_hooks:
            self.hook_interval = hook_interval
            self.hook_activated = True
            for name, module in self.hooked_modules.items():
                self.logs[name] = SummaryWriter(os.path.join(dir_path, name))
                self.hooksForward.append(module.register_forward_hook(self.__hook_forward))
                self.hooksBackward.append(module.register_backward_hook(self.__hook_backward))
        ELib.PASS()

    def __get_module_name(self, module):
        for name, cur_module in self.hooked_modules.items():
            if cur_module is module:
                return name
        return None

    def __hook_forward(self, module, input, output):
        if self.hook_activated and self.train_step % self.hook_interval == 0:
            name = self.__get_module_name(module)
            self.logs[name].add_histogram('activations', output.to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('weights', module.weight.data.to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('bias', module.bias.data.to('cpu').detach().numpy(), self.train_step)
        ELib.PASS()

    def __hook_backward(self, module, grad_input, grad_output):
        if self.hook_activated and self.train_step % self.hook_interval == 0:
            name = self.__get_module_name(module)
            self.logs[name].add_histogram('out-gradients', grad_output[0].to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('bias-gradients', grad_input[0].to('cpu').detach().numpy(), self.train_step)
        ELib.PASS()

    def close_logs(self):
        for cur_name, cur_log in self.logs.items():
            cur_log.close()
        for cur_h in self.hooksForward:
            cur_h.remove()
        for cur_h in self.hooksBackward:
            cur_h.remove()
        ELib.PASS()


class EBertClassifier(EClassifier):

    @staticmethod
    def create(class_type, training_object, config, **kwargs):
        result = class_type(config, **kwargs)
        try:
            print('loading modified pre-trained model..', end='. ', flush=True)
            state_dict = os.path.join(config.model_path, 'pytorch_model.bin')
            result.load_state_dict(torch.load(state_dict))
            print('loaded from ' + config.model_path, flush=True)
        except Exception as e:
            print('failed', flush=True)
            EBertClassifier.load_pretrained_bert_modules(result.__dict__['_modules'], config)
        # self.bert_classifier = BertForSequenceClassification.from_pretrained(
        #     self.config.model_path, config=self.config.bert_config)
        result._add_bert_hooks()
        result.training_object = training_object
        return result

    @staticmethod
    def load_pretrained_bert_modules(modules, config):
        mod_list = modules
        if type(modules) is not OrderedDict:
            mod_list = OrderedDict([('reconfig', modules)])
        loaded = set()
        for cur_module in mod_list.items():
            if type(cur_module[1]) is BertModel or isinstance(cur_module[1], EBertModelWrapper):
                print('{}: '.format(cur_module[0]), end='', flush=True)
                EBertClassifier.__load_pretrained_bert_module(cur_module[1], config, loaded)
            elif type(cur_module[1]) is nn.ModuleList:
                for c_ind, cur_child_module in enumerate(cur_module[1]):
                    if type(cur_child_module) is BertModel or isinstance(cur_child_module, EBertModelWrapper):
                        print('{}[{}]: '.format(cur_module[0], c_ind), end='', flush=True)
                        EBertClassifier.__load_pretrained_bert_module(cur_child_module, config, loaded)
        ELib.PASS()

    @staticmethod
    def __load_pretrained_bert_module(module, config, loaded):
        if type(module) is BertModel:
            if module not in loaded:
                module.load_state_dict(EBertClassifier.__load_pretrained_bert_layer(config).state_dict())
                if config.gradient_checkpointing:
                    module.gradient_checkpointing_enable()
                loaded.add(module)
            else:
                print('already loaded')
            ELib.PASS()
        elif isinstance(module, EBertModelWrapper):
            if module not in loaded:
                module.bert_layer.load_state_dict(EBertClassifier.__load_pretrained_bert_layer(config).state_dict())
                if config.gradient_checkpointing:
                    module.gradient_checkpointing_enable()
                loaded.add(module)
            else:
                print('already loaded')
            ELib.PASS()
        else:
            print(colored('unknown bert module to load', 'red'))

    @staticmethod
    def __load_pretrained_bert_layer(config):
        print('loading default model..', end='. ', flush=True)
        result = BertModel.from_pretrained(config.model_path, config=config.bert_config)
        print('loaded from ' + config.model_path, flush=True)
        return result

    def __init__(self):
        super(EBertClassifier, self).__init__()
        self.output_vecs = None
        self.output_vecs_detail = None
        self.training_object = None
        ELib.PASS()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.bert_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _add_bert_hooks(self):
        if len(self.hooked_modules) > 0:
            raise Exception('You have added hooks to EBertClassifier! \n'
                            'BERT was loaded from a file and hooks might have been overwritten! \n'
                            'override "_add_bert_hooks()" and add all of the hooks there!')

    def freeze_modules(self, m_list):
        for cur_m in m_list:
            self.training_object.remove_module_from_optimizer(cur_m)
        ELib.PASS()

    def unfreeze_modules(self, m_list):
        for cur_m in m_list:
            self.training_object.add_module_to_optimizer(cur_m)
        ELib.PASS()

    def set_modules_learning_rate(self, m_list, lr):
        for cur_m in m_list:
            self.training_object.set_module_learning_rate(cur_m, lr)
        ELib.PASS()


class EBertModelWrapper(EBertClassifier):

    def __init__(self, bert_config):
        super(EBertModelWrapper, self).__init__()
        self.bert_layer = BertModel(bert_config)

    def __format__(self, format_spec):
        ELib.PASS()


class EBertClassifierDACoordinated(EBertClassifier):

    class FType:
        none = 0
        pooled = 1
        output = 2
        both = 3

    def __init__(self, config, domain_count):
        super(EBertClassifierDACoordinated, self).__init__()
        self.config = config
        self.encoder_hidden_size = self.config.bert_config.hidden_size
        self.domain_count = domain_count
        self.active_domain = -1
        self.coral_scale = [0.0 for _ in range(domain_count)]
        self.stage = 1
        self.vectors = None
        self.clear_vectors()
        self.bert_states = list()
        self.cls_states = list()
        self.cls_shared_states = list()
        self.bert = nn.ModuleList()
        self.cls = nn.ModuleList()
        self.cls_shared = nn.ModuleList()
        for ind in range(self.domain_count):
            self.bert.append(BertModel(copy.deepcopy(self.config.bert_config)))
            self.cls.append(self.__get_cls_sequential())
            self.cls_shared.append(self.__get_cls_sequential())
        self.apply(self._init_weights)
        ## for grid search
        self.kl_scale = 1.0
        self.coral_density_balance = 0.0
        ##______________________________
        ELib.PASS()

    def __get_cls_sequential(self):
        return nn.Sequential(
            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, self.config.label_count)
        )

    def reconfig_encoders(self, scales, encoder_indices):
        print('reconfiguring the encoders ...')
        self.coral_scale = scales
        self.clear_domain_modules()
        while len(self.bert) > 0:
            del self.bert[0]
            del self.cls_shared[0]
        bert_dict = dict()
        for cur_ind in encoder_indices:
            if cur_ind not in bert_dict:
                bert_dict[cur_ind] = BertModel(copy.deepcopy(self.config.bert_config))
                self.cls_shared.append(self.__get_cls_sequential())
            self.bert.append(bert_dict[cur_ind])
        EBertClassifier.load_pretrained_bert_modules(self.bert, self.config)
        ELib.PASS()

    def clear_domain_modules(self):
        self.bert_states.clear()
        self.cls_states.clear()
        self.cls_shared_states.clear()

    def backup_domain_modules(self):
        self.clear_domain_modules()
        for ind in range(self.domain_count):
            self.bert_states.append(copy.deepcopy(self.bert[ind].state_dict()))
            self.cls_states.append(copy.deepcopy(self.cls[ind].state_dict()))
        for ind in range(len(self.cls_shared)):
            self.cls_shared_states.append(copy.deepcopy(self.cls_shared[ind].state_dict()))
        ELib.PASS()

    def restore_domain_modules(self):
        if len(self.bert_states) == 0:
            return
        for ind in range(self.domain_count):
            self.bert[ind].load_state_dict(self.bert_states[ind])
            self.cls[ind].load_state_dict(self.cls_states[ind])
        for ind in range(len(self.cls_shared)):
            self.cls_shared[ind].load_state_dict(self.cls_shared_states[ind])
        ELib.PASS()

    def clear_vectors(self):
        self.vectors = [{'src' : list(),
                         'tgt' : list()} for _ in range(self.domain_count)]
        ELib.PASS()

    def __feed(self, batch, bert_layer, cls_layer, ftype):
        b_output = bert_layer(batch['x'], attention_mask=batch['mask'], token_type_ids=batch['type'])
        output_pooled = b_output[1]
        result = None
        if ftype == self.FType.both or ftype == self.FType.output:
            result = cls_layer(output_pooled)
        if ftype == self.FType.pooled:
            return output_pooled
        elif ftype == self.FType.output:
            return result
        elif ftype == self.FType.both:
            return {'pooled': output_pooled, 'output': result}
        return None

    def __encoder(self, encoders, index):
        enc_set = set()
        for cur_enc in encoders:
            if cur_enc not in enc_set:
                enc_set.add(cur_enc)
            if len(enc_set) == index + 1:
                return cur_enc
        return None

    def forward(self, input_batch, apply_softmax):
        result = dict()
        d_c = self.domain_count
        if 'task_list' in input_batch:
            ## for testing
            result['logits'] = list()
            for ind in range(d_c):
                result['logits'].append(self.__feed(input_batch, self.bert[ind], self.cls[ind], self.FType.output))
            for ind in range(len(self.cls_shared)):
                cur_enc = self.__encoder(self.bert, ind)
                result['logits'].append(self.__feed(input_batch, cur_enc, self.cls_shared[ind], self.FType.output))
        else:
            ## for training
            if self.train_state == 1:
                with torch.no_grad():
                    result['src_pooled'] = list()
                    result['tgt_pooled'] = list()
                    for ind in range(d_c):
                        if len(input_batch[ind]) > 0:
                            result['src_pooled'].append(
                                self.__feed(input_batch[ind], self.bert[ind], self.cls[ind], self.FType.pooled))
                        else:
                            result['src_pooled'].append([])
                        if len(input_batch[d_c]) > 0:
                            result['tgt_pooled'].append(
                                self.__feed(input_batch[d_c], self.bert[ind], self.cls[ind], self.FType.pooled))
                        else:
                            result['tgt_pooled'].append([])
                ELib.PASS()
            elif self.train_state == 2:
                result['src_output'] = list()
                for ind in range(d_c):
                    result['src_output'].append(
                        self.__feed(input_batch[ind], self.bert[ind], self.cls[ind], self.FType.output))
                ELib.PASS()
            elif self.train_state == 3:
                result['src_pooled'] = list()
                result['src_output'] = list()
                result['tgt_pooled'] = list()
                for ind in range(d_c):
                    cur_src = self.__feed(input_batch[ind], self.bert[ind], self.cls[ind], self.FType.both)
                    result['src_pooled'].append(cur_src['pooled'])
                    result['src_output'].append(cur_src['output'])
                    result['tgt_pooled'].append(
                        self.__feed(input_batch[d_c], self.bert[ind], self.cls[ind], self.FType.pooled))
                ELib.PASS()
            elif self.train_state == 4:
                if self.stage == 1: # in case you later want to break this step to fit the gpu
                    ## cls + unbalanced coral + balanced coral + kl
                    self.training_object.delay_optimizer = False # in case you later want to break this step to fit the gpu
                    b0_ind = d_c # shared coral batch
                    b2_ind = d_c + 1 # KL batch
                    result['src_pooled'] = list()
                    result['src_output'] = list()
                    result['tgt_pooled_unbalanced'] = list()
                    result['tgt_pooled_balanced'] = list()
                    result['tgt_output_kl'] = list()
                    result['tgt_output_shared'] = list()
                    ## domain classfiers
                    for ind in range(d_c):
                        cur_src = self.__feed(input_batch[ind], self.bert[ind], self.cls[ind], self.FType.both)
                        result['src_pooled'].append(cur_src['pooled'])
                        result['src_output'].append(cur_src['output'])
                    ## target batches for coral
                    bert_coral_dict = dict()
                    for ind in range(d_c):
                        if self.bert[ind] not in bert_coral_dict:
                            bert_coral_dict[self.bert[ind]] = \
                                self.__feed(input_batch[b0_ind], self.bert[ind], self.cls[ind], self.FType.pooled)
                        result['tgt_pooled_unbalanced'].append(bert_coral_dict[self.bert[ind]])
                    ## target batch for KL
                    bert_kl_dict = dict()
                    for ind in range(d_c):
                        if self.bert[ind] not in bert_kl_dict:
                            bert_kl_dict[self.bert[ind]] = \
                                self.__feed(input_batch[b2_ind], self.bert[ind], self.cls[ind], self.FType.pooled)
                        output = self.cls[ind](bert_kl_dict[self.bert[ind]])
                        result['tgt_output_kl'].append(output)
                    ## target batch for the shared classifiers
                    for ind in range(len(self.cls_shared)):
                        cur_enc = self.__encoder(self.bert, ind)
                        output = self.cls_shared[ind](bert_kl_dict[cur_enc])
                        result['tgt_output_shared'].append(output)
                ELib.PASS()
        return result

