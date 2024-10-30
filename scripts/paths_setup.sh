path='/zhome/57/8/181461/thesis/lgvit/lgvit_repo'
model_path="${path}/models/deit_highway"

export PYTHONPATH=$path:$PYTHONPATH         # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path