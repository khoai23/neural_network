#!/bin/bash
#!/home/khoai23/environment/bin/python
# Show the commands ran
set -x

mail="quan94fm@gmail.com"
if [ "$#" -lt 4 ] || [ "$1" == "help" ]; then
  # launched extra comment
  echo "1 for mode, 2 for model, 3 for config file, 4 for run_dir. Automatically send an email containing log as an attachment."
  exit 1
fi
run_mode=$1
model=$2
config=$3
run_dir=$4
# mkdir just to be sure
mkdir -p $run_dir
# run the model with the run_dir
command="python -m uetnmt.bin.main $run_mode --model_type $model --config $config --run_dir $run_dir --gpu_allow_growth"
nohup $command >> $run_dir/temp.output 2>&1
if [ $? -eq 0 ]; then 
  status="Succeeded"
else
  status="Failed"
fi
date=`date '+%Hh%Mm%Ss %Y-%m-%d'`
# send the email with the log
content="to: $mail\nsubject: $status $date\n\nConfigurations:\n\tmode: $run_mode\n\tmodel: $model\n\tconfig: $config \n\trun_dir: $run_dir\n\tcommand: $command"
# vertical bar is pipe, and2 is combine
echo -e $content | (cat - && uuencode $run_dir/temp.output log.txt) | ssmtp $mail
# cleanup
rm $run_dir/temp.output
