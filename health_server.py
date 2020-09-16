#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/8/30 5:33 下午
# @Author : 郜志强
# @Version：V 0.1
# @File : test.py
# @desc :
import pandas as pd
import seaborn as sns; sns.set()
from easydict import EasyDict
from flask import Flask, make_response, request,json
from flask_cors import *
from common.logger import log
from common.response_status import ResponseStatus
from source.read_max_hr_p1 import split_text
from source.health_evaluation_p2 import predict_one
from source.health_prediction_p3 import predict_batch

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    data = EasyDict({'name': 'health_server', 'version': 1.0})
    return get_result_response(data)

@app.route('/readFile',methods=['POST'])
def readCSV():
    try:
        csv = request.files['file']
        data = pd.read_csv(csv,usecols=['liveHR'])
    except TypeError as te:
        log.error(te)
        return get_result_response(EasyDict({
            'code': ResponseStatus.TYPE_ERROR,
            'msg': 'The server received a csv error'
        }))
    except Exception as e:
        log.error(e)
        return get_result_response(EasyDict({
            'code': ResponseStatus.OTHER,
            'msg': 'The server receives a bad csv file'
        }))
    log.info("ip:{}".format(request.remote_addr))

    max_HR, str_max_HR, argmax = split_text(data['liveHR'])

    log.info("str_max_HR:{}".format(str_max_HR))
    return get_result_response(EasyDict({
        'code': ResponseStatus.SUCCESS,
        'msg': 'Success',
        'answer': str_max_HR,
    }))

@app.route('/single',methods=['POST'])
def single():
    try:
        data = json.loads(request.get_data())
        print(data)
        data['目标值'] = 0.0
        print(data)
        df = pd.DataFrame.from_dict(data,orient='index').T
        print(df)

    except TypeError as te:
        log.error(te)
        return get_result_response(EasyDict({
            'code': ResponseStatus.TYPE_ERROR,
            'msg': 'The server received a csv error'
        }))
    except Exception as e:
        log.error(e)
        return get_result_response(EasyDict({
            'code': ResponseStatus.OTHER,
            'msg': 'The server receives a bad csv file'
        }))
    log.info("ip:{}".format(request.remote_addr))
    answer = predict_one(df)
    log.info("answer:{}".format(answer))
    return get_result_response(EasyDict({
        'code': ResponseStatus.SUCCESS,
        'msg': 'Success',
        'answer': str(answer),
    }))

@app.route('/batch',methods=['POST'])
def batch():
    try:
        csv = request.files['file']
        df = pd.read_csv(csv)
    except TypeError as te:
        log.error(te)
        return get_result_response(EasyDict({
            'code': ResponseStatus.TYPE_ERROR,
            'msg': 'The server received a csv error'
        }))
    except Exception as e:
        log.error(e)
        return get_result_response(EasyDict({
            'code': ResponseStatus.OTHER,
            'msg': 'The server receives a bad csv file'
        }))
    log.info("ip:{}".format(request.remote_addr))
    prediction,label = predict_batch(df)

    log.info("prediction:{}".format(prediction))
    log.info("label:{}".format(label))
    return get_result_response(EasyDict({
        'code': ResponseStatus.SUCCESS,
        'msg': 'Success',
        'predict': prediction,
        'label': label,
    }))


def get_result_response(msg):
    response = make_response(msg)
    response.headers["Content-Type"] = "application/json"
    response.headers["name"] = "health_server"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9003)


