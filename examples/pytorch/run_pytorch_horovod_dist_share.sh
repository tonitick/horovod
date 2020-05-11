profdir=/data2/home/zzhong/profdatashare
interval="0.1"
iters="3"
ngpu="1"
nproc=$(($ngpu + $ngpu))
netif="eno1"
# for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
# for model in vgg11 vgg13 vgg16 vgg19 resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
for model in resnet50 resnet34
do
	mkdir -p ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/throughput

    export CUDA_VISIBLE_DEVICES=0
	mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 > ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/throughput/${model}.txt &
	pid1=$!
	echo "pid1=${pid1}"
	
    export CUDA_VISIBLE_DEVICES=1
    mpirun --mca oob_tcp_if_include ${netif} --mca btl_tcp_if_include ${netif} -np ${nproc} -H proj54:${ngpu},proj55:${ngpu} -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=${netif} -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model --batch-size 32 > ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/throughput/${model}.txt &
	pid2=$!
	echo "pid2=${pid2}"

	python cluster_monitor/collectl.py -i ${interval} --start -o ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/collectl/${model} -m cluster_monitor/workers
	python cluster_monitor/top.py -i ${interval} --start -o ${profdir}/result_pytorch_horovod_dist_${ngpu}gpupm_${nproc}proc_${interval}/top/${model} -m cluster_monitor/workers
	
	wait $pid1
	wait $pid2

	python cluster_monitor/collectl.py --stop -m cluster_monitor/workers
	python cluster_monitor/top.py --stop -m cluster_monitor/workers
done
