import kerastuner

class MyTuner(kerastuner.tuners.BayesianOptimization):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1, 64)
    kwargs['epochs'] = 500 #trial.hyperparameters.Int('epochs', 10, 30)
    super(MyTuner, self).run_trial(trial, *args, **kwargs)
