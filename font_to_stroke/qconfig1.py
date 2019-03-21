def main():
    # print("start")
    config = dict()
    config['seed'] = 17

    config['training_data'] = './data/deepwriting_training.npz'
    config['validation_data'] = './data/deepwriting_validation.npz'
    config['validate_model'] = False

    config['model_save_dir'] = './runs/'

    config['checkpoint_every_step'] = 1000
    config['validate_every_step'] = 25  # validation performance
    # Model predictions are converted into images and displayed in Tensorboard. Set 0 to disable image summaries.
    config['img_summary_every_step'] = 0
    config['print_every_step'] = 2  # print

    config['reduce_loss'] = "mean_per_step"  # "mean" "sum_mean", "mean", "sum".
    config['batch_size'] = 64
    config['num_epochs'] = 1000
    config['hidden_size'] = 900
    config['learning_rate'] = 1e-5
    config['draw'] = 10
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_rate'] = 0.96
    config['num_mixture_components'] = 20

    config['create_timeline'] = False
    config['tensorboard_verbose'] = 0  # 1 for histogram summaries and 2 for latent space norms.
    config['use_dynamic_rnn'] = True
    config['use_bucket_feeder'] = True
    config['use_staging_area'] = True

    config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
    config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.

    config['vrnn_cell_cls'] = 'HandWritingVRNNGmmCell'
    config['model_cls'] = 'HandwritingVRNNGmmModel'
    config['dataset_cls'] = 'HandWritingDatasetConditionalTF'

    #
    # VRNN Cell settings
    #
    config['output'] = {}
    config['output']['keys'] = ['out_mu', 'out_sigma', 'out_rho', 'out_pen', 'out_eoc']
    config['output']['dims'] = [2, 2, 1, 1, 1]  # Ideally these should be set by the model.
    config['output']['activation_funcs'] = [None, 'softplus', 'tanh', 'sigmoid', 'sigmoid']

    config['latent_rnn'] = {}                       # See get_rnn_cell function in tf_model_utils.
    config['latent_rnn']['num_layers'] = 1          # (default: 1)
    config['latent_rnn']['cell_type'] = 'lstm'       # (default: 'lstm')
    config['latent_rnn']['size'] = 512              # (default: 512)

    # Pass None if you want to use fully connected layers in the input or output layers.
    config['input_rnn'] = {}
    if config['input_rnn'] == {}:
        config['input_rnn']['num_layers'] = 1
        config['input_rnn']['cell_type'] = 'lstm'
        config['input_rnn']['size'] = 512

    return config
