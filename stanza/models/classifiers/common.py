import logging

logger = logging.getLogger('stanza')

def load_elmo(elmo_options_path, elmo_weights_path, saved_scalar_mixes=None):
    # This import is here so that Elmo integration can be treated
    # as an optional feature
    import allennlp.modules.elmo as elmo

    logger.info("Loading elmo: options %s weights %s" % (elmo_options_path, elmo_weights_path))
    if saved_scalar_mixes:
        elmo_model = elmo.Elmo(elmo_options_path, elmo_weights_path, 1,
                               scalar_mix_parameters=saved_scalar_mixes)
    else:
        elmo_model = elmo.Elmo(elmo_options_path, elmo_weights_path, 1)
    logger.info("Finished loading elmo")
    return elmo_model

