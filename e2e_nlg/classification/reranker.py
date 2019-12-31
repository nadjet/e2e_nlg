from e2e_nlg.formatting.mrs_formatter import MR_Formatter
from e2e_nlg.formatting.relexicalizer import ReLexicalizer
from utils.seq2seq_predict import PredictUtils
from fastai.text import DatasetType
from utils.log import logger
import os


class Reranker:
    def __init__(self, mr_predictor, predict_utils, k=10, p=0.2, T=0.5):
        '''
               :param predict_utils: the Predictor object
               :param mr_predictor: the Classifier object
               :param k: the maximum number of outputs to be returned
               :param p: the top-p sampling probability
        '''
        self.predict_utils = predict_utils
        self.mr_predictor = mr_predictor
        self.k = k
        self.p = p
        self.T =  T
        logger.info("p={}, k={}, T={}".format(self.p,self.k,self.T))


    @staticmethod
    def check_predictions(predictions,is_input=False):
        new_predictions = []
        has_eat_type = False
        for item in predictions:
            item = item.replace("less than £20", "1")
            item = item.replace("more than £30", "3")
            item = item.replace("£20-25", "2")
            item = item.replace("1 out of 5","1")
            item = item.replace("3 out of 5","2")
            item = item.replace("5 out of 5","3")
            item = item.replace("low","1")
            item = item.replace("average","2")
            item = item.replace("moderate","2")
            item = item.replace("cheap","1")
            item = item.replace("high","3")
            new_predictions.append(item)
            if "eatType" in item:
                has_eat_type = True
        if not has_eat_type and is_input:
            new_predictions.append("eatType[restaurant]")
        return new_predictions

    def get_f1(self, text, input_predictions):
        output_predictions = self.mr_predictor.predict(ReLexicalizer.clean_str(str(text)))
        output_predictions = (str(output_predictions[0])).split(";")
        output_predictions = Reranker.check_predictions(output_predictions)
        f1 = PredictUtils.calculate_fscore(input_predictions, output_predictions)
        return f1

    def get_input_candidates(self, input_preds, x, y):
        '''
        :param inp_preds: the input mrs
        :param x: the input tensors
        :param y: the output tensors
        :return: output candidates
        '''
        candidates, f1s = [], []
        num_iterations = 0
        identical = 0
        while len(candidates)<self.k and num_iterations < self.k*2 and identical < self.k:
            num_iterations = num_iterations + 1
            inp, target, output = self.predict_utils.predict_with_decode(x, y, p = self.p, T=self.T)
            if output in candidates:
                identical = identical + 1
                continue
            f1 = self.get_f1(output, input_preds)
            candidates.append(output)
            f1s.append(f1)
        return candidates, f1s

    def write_inputs_candidates(self, file_name, xs, ys, zs, rxs, rys, rzs):
        with open(file_name, "w") as f:
            w = writer(f,delimiter="\t")
            w.writerow(["group","output", "f1","greedy","input","target"])
            current_input = ""
            counter=1
            for idx in range(len(rxs)):
                print(idx)
                rx, ry, rz = rxs[idx], rys[idx], rzs[idx]
                x, y, z = xs[idx], ys[idx], zs[idx]
                mr_formatter = MR_Formatter(str(rx))
                mr_formatter.set_attributes()
                mr_formatter.attributes.remove('name[XXX]')
                input_preds = mr_formatter.attributes
                input_preds = Reranker.check_predictions(input_preds,is_input=True)
                outputs, f1s = self.get_input_candidates(input_preds, x, y)
                for i in range(len(outputs)):
                    if str(rx)!=current_input:
                        counter=counter+1
                    w.writerow([counter,str(outputs[i]), str(f1s[i]), str(rz),str(rx), str(ry)])
                if idx==2:
                    break


    def rerank(self, inp, outp, x, y):
        '''
        :param inp: the input
        :param outp: the output of greedy search
        :param x: the input tensor
        :param y: the output tensor
        :return: output of neural sampling with highest f1 wrt input
        '''
        mr_formatter = MR_Formatter(str(inp))
        mr_formatter.set_attributes()
        mr_formatter.attributes.remove('name[XXX]')
        input_preds = mr_formatter.attributes
        input_preds = Reranker.check_predictions(input_preds)
        f1 = self.get_f1(outp, input_preds)
        if f1==1:
            return outp,f1
        outputs, f1s  = self.get_input_candidates(input_preds,x, y)
        max_f1 = f1
        max_output = outp
        for i in range(len(outputs)):
            if f1s[i]>max_f1:
                max_output=outputs[i]
                max_f1=f1s[i]
        selected_output = max_output
        return selected_output,max_f1

    def rerank_generator(self, xs, ys, zs, rxs, rys, rzs):
        counter=0
        for idx in range(len(rxs)):
            rx, ry, rz = rxs[idx], rys[idx], rzs[idx]
            x, y, z = xs[idx], ys[idx], zs[idx]
            best_output,f1 = self.rerank(rx, rz, x, y)
            best_output = str(best_output)
            if str(rz)!=best_output:
                counter=counter+1
            logger.info("{} {}".format(counter,idx))
            if best_output!=str(rz):
                yield [1,str(best_output),f1, str(rz),str(rx),str(ry)]
            else:
                yield [0, str(best_output), f1, str(rz),str(rx), str(ry)]


import sys
from e2e_nlg.loading.loader import  E2ENLGDataLoader
from e2e_nlg.classification.mr_predictor import MrPredictor
from fastai.text import load_learner
from csv import writer

def do_reranking(reranker,file_name,xs,ys,zs,rxs,rys,rzs):
    with open(file_name,"w") as f:
        w = writer(f,delimiter ='\t')
        w.writerow(["different","output","f1","old_output","input","target"])
        for row in reranker.rerank_generator(xs, ys, zs, rxs, rys, rzs):
            w.writerow(row)


if __name__ == "__main__":
    main_path = sys.argv[1]
    fasttext_path = sys.argv[2]
    dataset_path = sys.argv[3]
    out_folder = sys.argv[4]

    dl = E2ENLGDataLoader(dataset_path, "trainset.csv", "devset.csv", percentile=100)
    dl.setDataAndMaxSize(bs=32)
    data = dl.data

    # seq2seq model
    learn = load_learner(os.path.join(main_path,"models"))
    #dl.load_data(os.path.join(main_path,"models"))
    learn.data = dl.data

    predictor = MrPredictor(os.path.join(main_path,"models","classifier"), dataset_path, "trainset.csv", "devset.csv", "testset_w_refs.csv")
    predict_utils = PredictUtils(learn)

    reranker = Reranker(predictor,predict_utils, k=20, p=0.2)

    rxs, rys, rzs, xs, ys, zs = predict_utils.preds_acts(ds_type=DatasetType.Valid)

    #reranker.write_inputs_candidates("candidates.csv",xs, ys, zs, rxs, rys, rzs)

    do_reranking(reranker,"reranker.csv",xs, ys, zs, rxs, rys, rzs)