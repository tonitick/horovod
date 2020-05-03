interval="0.1"
profdir=/data2/home/zzhong/profdata
iters="200"
ngpu="4"
for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
# for model in resnet50
do
	export CUDA_VISIBLE_DEVICES=0,1,2,3
	mkdir -p ${profdir}/result_pytorch_horovod_single_${ngpu}gpu_${interval}/throughput
	mpirun --allow-run-as-root -np ${ngpu} -H localhost:${ngpu} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python pytorch_horovod_benchmark.py --num-iters ${iters} --model=$model > ${profdir}/result_pytorch_horovod_single_${ngpu}gpu_${interval}/throughput/${model}.txt &
	python cluster_monitor/top.py -i ${interval} --start -o ${profdir}/result_pytorch_horovod_single_${ngpu}gpu_${interval}/top/${model} -m cluster_monitor/worker

	wait $!

	python cluster_monitor/top.py --stop -m cluster_monitor/worker
done
