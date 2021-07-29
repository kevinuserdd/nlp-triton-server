import numpy as np
import sys
import json
from torch import nn
from transformers import BertTokenizer
import pickle
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class event_postprocess(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(event_postprocess, self).__init__()
        self.init_utils()

    def init_utils(self):
        self.params = self.load_json('/utils/param.json')
        self.tag2id = self.load_json(self.params['tag2id'])
        self.id2label = dict([(v, k) for k, v in self.tag2id.items()])
        event_schema = self.load_data(self.params["schema_dir"])
        schema_datas = self.read_by_lines(self.params["schema_dir"])
        self.schema = self.read_schema(schema_datas)
        label2idx = self.label2index(event_schema, flag=True)
        self.idx2label = {label2idx[label] : label for label in label2idx.keys()}
    
    def read_by_lines(self,path):
        result = list()
        with open(path, "r", encoding="utf-8") as infile:
            for line in infile:
                result.append(line.strip())
        return result

    def read_schema(self,schema_datas):
        schema = {}
        for s in schema_datas:
            d_json = json.loads(s)
            schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
        return schema

    def extract_result(self,text, labels):
        ret, is_start, cur_type = [], False, None
        if len(text) != len(labels):
            labels = labels[:len(text)]
        for i, label in enumerate(labels):
            if label != u"O":
                 _type = label.split('-')[-1]
                 if label.startswith(u"B-"):
                     is_start = True
                     cur_type = _type
                     ret.append({"start": i, "text": [text[i]], "type": _type})
                 elif _type != cur_type:
                     cur_type = _type
                     is_start = True
                     ret.append({"start": i, "text": [text[i]], "type": _type})
                 elif is_start:
                     ret[-1]["text"].append(text[i])
                 else:
                     cur_type = None
                     is_start = False
            else:
                 cur_type = None
                 is_start = False
        return ret

    def label2index(self,event_schema, flag=True):
        arguments = ['O']
        tags = ['B', 'I', 'S']
        if flag:
            for t in tags:
                arguments.append(t + '-' + '时间')
     
        for event in event_schema:
            for role in event['role_list']:
                if flag and role['role'] == '时间':
                    continue
                for t in tags:
                    arguments.append(t + '-' + event['event_type'] + '-' + role['role'])
        label2index = {}
        for i, argument in enumerate(arguments):
            label2index[argument] = i
        return label2index

    def load_data(self,dataset_dir):
        file = open(dataset_dir, 'r', encoding='utf-8')
        dataset = []
        for line in file.readlines():
            dataset.append(json.loads(line))
        return dataset
            
    def load_json(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
            
    def trans2label(self,id2tag,data,lengths):
        new = []
        for i,line in enumerate(data):
            tmp = [id2tag[word] for word in line]
            tmp = tmp[:lengths[i]]
            new.append(tmp)
        return new

    def forward(self,logits,role_logits,lengths,text):
        """
        input0:np.array((bs,seq_len))
        """
        #trigger
        texts = []
        for m in text.as_numpy():
            texts.append(m[0].decode('utf-8'))
        scores = logits.as_numpy()
        res = []
        for batch in scores:
            batch_label = []
            for idx, class_probability in enumerate(batch):
                if class_probability > 0.5:
                    batch_label.append(self.id2label[idx])
            res.append(batch_label)
        role_scores = np.argmax(role_logits.as_numpy(),axis = -1)
        print(lengths.as_numpy())
        preds = self.trans2label(self.idx2label,role_scores,lengths.as_numpy().squeeze(1))
        #process role
        sent_role_mapping = []
        pred_ret = []
        for d in zip(texts,preds):
            r_ret = self.extract_result(d[0],d[1])
            role_ret = {}
            for r in r_ret:
                 role_type = r["type"]
                 if role_type not in role_ret:
                     role_ret[role_type] = []
                 role_ret[role_type].append("".join(r["text"]))
            sent_role_mapping.append(role_ret)
        #process trigger
        for role_map,pred_trigger_types in zip(sent_role_mapping,res):
            event_list = []
            for event_type in pred_trigger_types:
                  role_list = self.schema[event_type]
                  arguments = []
                  for role_type, ags in role_map.items():
                      if role_type not in role_list:
                          continue
                      for arg in ags:
                          if len(arg) == 1:
                              continue
                          arguments.append({"role": role_type, "argument": arg})
                  event = {"event_type": event_type, "arguments": arguments}
                  event_list.append(event)
            pred_ret.append({"event_list": event_list})
       # pred_tags.extend([[self.idx2label.get(idx) for idx in indices] for indices in outputs])
        dump_entities = np.array(pred_ret,dtype=object)
        return dump_entities

        ###
        #entities = np.array(entities,dtype=np.object_)
        #entities = entities[:,np.newaxis]

        #return entities


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "EVENT_LABEL_OUTPUT")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
                             output0_config['data_type'])

        # Instantiate the PyTorch model
        self.model = event_postprocess()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        print('Starting execute')
        output0_dtype = self.output0_dtype
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "TRIGGER_OUTPUT_TENSOR")
            in_1 = pb_utils.get_input_tensor_by_name(request, "ROLE_OUTPUT_TENSOR")
            in_2 = pb_utils.get_input_tensor_by_name(request, "LENGTHS")
            in_3 = pb_utils.get_input_tensor_by_name(request, "TEXT")
            out_0  = self.model(in_0,in_1,in_2,in_3)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("EVENT_LABEL_OUTPUT",
                                           out_0.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])

            #inference_response = pb_utils.InferenceResponse(
            #    output_tensors=[out_tensor_0,out_tensor_1,out_tensor_2])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
