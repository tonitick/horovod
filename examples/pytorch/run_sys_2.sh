for model in vgg16 resnet50 resnet101 resnet152 densenet121 densenet201 densenet169 
do
	export CUDA_VISIBLE_DEVICES=0,2
	mpirun --allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters 100 --model=$model > result_sys_2/${model}_1.txt &
	export CUDA_VISIBLE_DEVICES=1,3

	mpirun --allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters 100 --model=$model > result_sys_2/${model}_2.txt

	wait
done
