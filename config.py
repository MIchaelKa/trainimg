
class GlobalConfig:
    target_size = 11
    target_columns = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    learning_rate = 2e-4
    scheduler_batch_update = False
