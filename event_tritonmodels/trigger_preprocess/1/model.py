import numpy as np
import sys
import json
from torch import nn
from transformers import BertTokenizer

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class trigger_preprocess(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(trigger_preprocess, self).__init__()
        self.init_utils()

    def init_utils(self):
        self.params = self.load_json('/utils/param.json')
        self.tokenizer = BertTokenizer.from_pretrained(self.params['pretrained_model'])
        self.label2id = self.load_json(self.params['tag2id'])
        self.id2label = {_id: _label for _label, _id in list(self.label2id.items())}

    def load_json(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def gen_features(self,tokens, tokenizer, max_len):
        input_ids, tags, masks = [], [], []
        for i, token in enumerate(tokens):
            if len(token) >= max_len - 1:
                token = token[0:max_len - 1]
            mask = [1] * len(token)
            token = '[CLS] ' + ' '.join(token)
            tokenized_text = tokenizer.tokenize(token)
            input_id = tokenizer.convert_tokens_to_ids(tokenized_text)
            mask = [0] + mask
            # padding
            if len(input_id) < max_len:
                input_id = input_id + [0] * (max_len - len(input_id))
                mask = mask + [0] * (max_len - len(mask))

            assert len(input_id) == max_len
            assert len(mask) == max_len

            input_ids.append(input_id)
            masks.append(mask)
        return input_ids, masks

    def forward(self, input0):
        """
        input0:np.array((bs,seq_len))
        """

        sentences = input0
        data = []
        for sentence in sentences:
            data.append(sentence)
        input_ids, masks = self.gen_features(data,self.tokenizer,384)
        return np.array(input_ids),np.array(masks)


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
            model_config, "INPUT_IDS")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "MASKS")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        # Instantiate the PyTorch model
        self.model = trigger_preprocess()

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
        output1_dtype = self.output1_dtype

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "RAW_TEXT")
            inputs = []
            for instance in  in_0.as_numpy():
                sen = instance[0].decode('utf-8')
                #inputs.append(str(instance[0]))
                inputs.append(sen)
            out_0 ,out_1 = self.model(inputs)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("INPUT_IDS",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("MASKS",
                                           out_1.astype(output1_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0,out_tensor_1])

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
