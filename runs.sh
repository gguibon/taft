# Our model is TAFT which possess an adaptive layer and an asynchronous fine-tuning as explained in the paper (see Figure 1)

# Some inline comments refers to parts of the paper

# For all these commands, you can change the dataset with the --dataset option. 
# The default one is the dummy dataset which mimics our confidential dataset structure with gibbering texts and randomly assigned labels.
# This dummy dataset do not achieve good results of course, but allows you to run the model, given this legal restriction we have.

# The code is tested on GPU only. Once a training is done, you can open 'lightning_logs' directory to see the model, plots, reports and tensorboard files.


# ==== TAFT: Task-Adaptive Fine-Tuning : our model that allows domain and task adaptation at once using an adaptive layer ====

## Step 1
CUDA_VISIBLE_DEVICES=0 python3.8 speaker_role_pretraining.py

bestmodelpath=$(head -n 1 temp.txt)

## Step 2 : PC (Problematic Conversation)
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --status pc

## Step 2 : Problem Status
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --status status



# ==== TAFT noAdapt : no adaptive layer ====

## Step 1 : Speaker role identification on unlabeled data
CUDA_VISIBLE_DEVICES=0 python3.8 speaker_role_pretraining.py --type direct

bestmodelpath=$(head -n 1 temp.txt)

## Step 2 : PC (Problematic Conversations) on labeled data
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --type direct --status pc

## Step 2 : Problem Status on labeled data
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --type direct --status status


# ==== TAPT: Task-Adaptive Pre-Training ====

## Step 1 : Speaker role identification on unlabeled data
CUDA_VISIBLE_DEVICES=0 python3.8 speaker_role_pretraining.py --type direct

bestmodelpath=$(head -n 1 temp.txt)

## Step 2 : PC (Problematic Conversations) on labeled data
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --type tapt --status pc

## Step 2 : Problem Status on labeled data
CUDA_VISIBLE_DEVICES=0 python3.8 taft_target_finetuning.py --modeltoload $bestmodelpath --type tapt --status status


# ==== DAPT: Domain-Adaptive Pre-Training ====
## Step 1
CUDA_VISIBLE_DEVICES=0 python3.8 dapt_mlm.py

## Step 2
best_checkpoint=results/checkpoint-100/pytorch_model.bin #put the best model, for instance: results/checkpoint-100/pytorch_model.bin 
### applied to problem status
CUDA_VISIBLE_DEVICES=0 python3.8 dapt_status_ft.py --modeltoload $best_checkpoint --status status
### applied to problematic conversation identification
CUDA_VISIBLE_DEVICES=0 python3.8 dapt_status_ft.py --modeltoload $best_checkpoint --status pc
### applied to satisfaction prediction
CUDA_VISIBLE_DEVICES=0 python3.8 dapt_satisfaction_ft.py --modeltoload $best_checkpoint


# Some exploration about training information for all the above models
# lightning_logs directory (created after training) also contains saved model weights, plots, and reports
tensorboard --logdir=lightning_logs