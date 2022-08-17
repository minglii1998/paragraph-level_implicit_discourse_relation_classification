temp_config_model_name = 'sa'
temp_config_data_path = 'data/t5_embedding_large.pt'
temp_config_logger = 'large_t5_sa'

# temp_config_model_name = 'ori'
# temp_config_data_path = 'data/t5_embedding_large.pt'
# temp_config_logger = 'large_t5_ori'

logger_name = 'logs_mixed_all'

# used for mixed model configration
extra_mixed_conf = {}
extra_mixed_conf['mixed_vec_path'] = 'LiMNet_pdtb_vec.pt' 
# LiMNet_pdtb_vec, LiMNet_pdtb_vec_from_all
extra_mixed_conf['where_to_add'] = 'before_classification' # only choice now
extra_mixed_conf['add_type'] = 'pred' 
# local_sentence_embedding, global_sentence_shift, mixed_sentence_embedding_1, mixed_sentence_embedding_2, pred, one_hot
extra_mixed_conf['extra_ffn_dim'] = 0 # 0, 300, 512, 1024

if len(extra_mixed_conf) != 0:
    for k in extra_mixed_conf.keys():
        temp_config_logger = temp_config_logger + str(extra_mixed_conf[k]) + '.'
