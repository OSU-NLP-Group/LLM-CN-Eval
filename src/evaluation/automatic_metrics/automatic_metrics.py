import pandas as pd 
import numpy as np
import evaluate
import bart_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Uses NLTK sentence_bleu to generate bleu1, bleu3, and bleu4 scores for all candidates from a generation model
def bleu(candidates_df, reference_df, generation_model):
    bleu1_scores, bleu3_scores, bleu4_scores = [], [], []
    smooth = SmoothingFunction().method1
    for i in range(0, len(candidates_df)):
        candidates = candidates_df.iloc[i]
        reference = [reference_df.iloc[i].split()]
        bleu1_scores.append(sentence_bleu(reference, candidates[generation_model].split(), weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu3_scores.append(sentence_bleu(reference, candidates[generation_model].split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu(reference, candidates[generation_model].split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    return bleu1_scores, bleu3_scores, bleu4_scores

# Use rouge_score to generate ROUGE scores for all candidates from a generation model 
def rouge(candidates_df, reference_df, generation_model):
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    for i in range(0, len(candidates_df)):
        candidates = candidates_df.iloc[i]
        reference = reference_df.iloc[i]
        score = scorer.score(reference, candidates[generation_model])
        rouge_scores.append(score['rougeL'][2])
    return rouge_scores

# Uses evaluate module to generate METEOR scores for all candidates from a generation model 
def meteor(candidates_df, reference_df, generation_model):
    meteor_scores = []
    scorer = evaluate.load('meteor')
    for i in range(0, len(candidates_df)):
        candidates = candidates_df.iloc[i]
        reference = [reference_df[i]]
        score = scorer.compute(predictions=[candidates[generation_model]], references=reference)
        meteor_scores.append(score['meteor'])
    return meteor_scores

# Uses evaluate module to generate BERTScore for all candidates from a generation model 
def bertscore(candidates_df, reference_df, generation_model):
    bert_scores = []
    scorer = evaluate.load('bertscore')
    for i in range(0, len(candidates_df)):
        candidates = candidates_df.iloc[i]
        reference = [reference_df[i]]
        score = scorer.compute(predictions=[candidates[generation_model]], references=reference, lang='en', model_type='roberta-large')
        bert_scores.append(score['f1'][0])
    return bert_scores

# Uses bart_score.py to generate BARTScore for all candidates from a generation model 
def bartscore(candidates_df, reference_df, generation_model, bart_type):
    bart_scores_precision, bart_scores_recall, bart_scores_f1 = [], [], []

    # Defines whether BART, BART finetuned on CNN Daily Mail data, or BART finetuned on CNN Daily Mail and ParaBank is being used
    if bart_type == 'base':
        scorer = bart_score.BARTScorer(checkpoint='facebook/bart-large')
    elif bart_type == 'cnn':
        scorer = bart_score.BARTScorer(checkpoint='facebook/bart-large-cnn')
    elif bart_type == 'cnn-para':
        scorer = bart_score.BARTScorer(checkpoint='facebook/bart-large-cnn')
        scorer.load(path='bart_score.pth')

    for i in range(0, len(candidates_df)):
        candidates = candidates_df.iloc[i]
        reference = [reference_df[i]]

        # Computes BARTScore for precision, recall, and F1
        precision_score = scorer.score(tgts=[candidates[generation_model]], srcs=reference)
        bart_scores_precision.append(precision_score[0])

        recall_score = scorer.score(tgts=reference, srcs=[candidates[generation_model]])
        bart_scores_recall.append(recall_score[0])

        f1_score = (recall_score[0] + precision_score[0]) / 2
        bart_scores_f1.append(f1_score)
    return bart_scores_precision, bart_scores_recall, bart_scores_f1

candidates_df = pd.read_csv("../all_candidateCNs.csv")
reference_df = candidates_df['COUNTER_NARRATIVE']

generation_model = "dialoGPT" # generation_model must be dialoGPT, chatGPT, or vicuna
output_df = pd.DataFrame(columns=['HATE_SPEECH', 'COUNTER_NARRATIVE', 'TARGET', generation_model, 'bleu1', 'bleu3', 'bleu4', 
                    'rouge', 'meteor', 'bertscore', 'bartscore_precision', 'bartscore_cnn_precision', 
                    'bartscore_cnn_para_precision', 'bartscore_recall', 'bartscore_cnn_recall', 'bartscore_cnn_para_recall',
                    'bartscore_f1', 'bartscore_cnn_f1', 'bartscore_cnn_para_f1'])
output_df['COUNTER_NARRATIVE'] = reference_df
output_df['HATE_SPEECH'] = candidates_df['HATE_SPEECH']
output_df['TARGET'] = candidates_df['TARGET']
output_df[generation_model] = candidates_df[generation_model]
candidates_df = candidates_df.drop(["HATE_SPEECH", "TARGET", "COUNTER_NARRATIVE"], axis=1)

bleu1_scores, bleu3_scores, bleu4_scores = bleu(candidates_df, reference_df, generation_model)
output_df['bleu1'] = bleu1_scores
output_df['bleu3'] = bleu3_scores
output_df['bleu4'] = bleu4_scores

rouge_scores = rouge(candidates_df, reference_df, generation_model)
output_df['rouge'] = rouge_scores

meteor_scores = meteor(candidates_df, reference_df, generation_model)
output_df['meteor'] = meteor_scores

bert_scores = bertscore(candidates_df, reference_df, generation_model)
output_df['bertscore'] = bert_scores

bart_scores_precision, bart_scores_recall, bart_scores_f1 = bartscore(candidates_df, reference_df, generation_model, bart_type='base')
output_df['bartscore_precision'] = bart_scores_precision
output_df['bartscore_recall'] = bart_scores_recall
output_df['bartscore_f1'] = bart_scores_f1

bart_scores_precision, bart_scores_recall, bart_scores_f1 = bartscore(candidates_df, reference_df, generation_model, bart_type='cnn')
output_df['bartscore_cnn_precision'] = bart_scores_precision
output_df['bartscore_cnn_recall'] = bart_scores_recall
output_df['bartscore_cnn_f1'] = bart_scores_f1

bart_scores_precision, bart_scores_recall, bart_scores_f1 = bartscore(candidates_df, reference_df, generation_model, bart_type='cnn_para')
output_df['bartscore_cnn_para_precision'] = bart_scores_precision
output_df['bartscore_cnn_para_recall'] = bart_scores_recall
output_df['bartscore_cnn_para_f1'] = bart_scores_f1

output_df.to_csv(generation_model + "_scores_auto.csv", index=False)