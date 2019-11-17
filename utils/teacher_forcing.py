from fastai.text import LearnerCallback

class TeacherForcing(LearnerCallback):

    def __init__(self, learn, end_epoch):
        super().__init__(learn)
        self.end_epoch = end_epoch

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train: return {'last_input': [last_input, last_target]}

    def on_epoch_begin(self, epoch, **kwargs):
        self.learn.model.pr_force = 1 - epoch / self.end_epoch