import re
import random
from PIL import ImageFilter

def pre_text(question, max_ques_words, max_length=200):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('–', ' ')
    question = question.strip('\n')
    question = question.rstrip(' ')

    question_words = question.split(' ')
    words = []
    count = 0 
    leng = 0
    for word in question_words:
        ans = re.match( r"([a-z0-9]+)", word)
        if ans is None or len(ans.group(0))<len(word):
            continue
        leng += len(word)+1
        if count >= max_ques_words or leng>=max_length:
            break
        words.append(word)
        count += 1
    question = ' '.join(words)
    return question


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x