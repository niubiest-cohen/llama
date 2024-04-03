set -e

rm -f *.bin
rm -f `ls *.txt | grep -v 'requirements.txt'` 


rm -rf golden/
mkdir golden


torchrun --nproc_per_node 1 example_text_completion.py \
  --ckpt_dir ./Llama-2-7b/ \
  --tokenizer_path ./Llama-2-7b/tokenizer.model \
  --max_seq_len 512 \
  --max_batch_size 1 \
  --max_gen_len 200 
# --master_port 25641 
