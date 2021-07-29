from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np
from datetime import datetime
model_name = "role_ensemble"

with grpcclient.InferenceServerClient("localhost:11501") as client:

    with open('./event_testdata.txt','r',encoding = 'utf-8')as f:
        data = f.readlines()

    lines= []
    for line in data:
        line = line.replace('\n','')
        line = line.split('\t')[-1]
        lines.append(line)
    n = len(lines)
    input0_data = np.array(lines).astype(np.object_)
    input0_data = input0_data.reshape((n,-1))
    input0_data = input0_data[:32]

    inputs = [
    grpcclient.InferInput("TEXT", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype))
    ]
    inputs[0].set_data_from_numpy(input0_data)
    outputs = [
        grpcclient.InferRequestedOutput("event_list"),
    ]
    start = datetime.now()
    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)
    end = datetime.now()
    result = response.get_response()
    res = [i.decode('utf-8') for i in response.as_numpy("event_list")]
    print(res)
    #print("INPUT0 ({}), '\n',OUTPUT0 ({}),'\n',result {}".format(
    #    input0_data, response.as_numpy("event_list"),result))
