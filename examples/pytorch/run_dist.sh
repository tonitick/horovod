# for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
for model in resnet50
do
	export CUDA_VISIBLE_DEVICES=0,1
	mpirun --mca oob_tcp_if_include eno1 --mca btl_tcp_if_include eno1  -np 4 -H proj54:2,proj55:2 -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=eno1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters 1 --model=$model --batch-size 32 > result_dist/${model}.txt &
	
	python cluster_monitor/collectl.py -i 0.1 --start -o $(pwd)/cluster_monitor/collectl/${model} -m cluster_monitor/workers
	python cluster_monitor/top.py -i 0.1 --start -o $(pwd)/cluster_monitor/top/${model} -m cluster_monitor/workers
	
	wait $!

	python cluster_monitor/collectl.py --stop -m cluster_monitor/workers &
	python cluster_monitor/top.py --stop -m cluster_monitor/workers
done
