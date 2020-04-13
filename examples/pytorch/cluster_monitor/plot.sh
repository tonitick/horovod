for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn resnet18 resnet34 resnet50 resnet101 resnet152 squeezenet1_0 squeezenet1_1 densenet121 densenet169 densenet161 densenet201
do
	python plot.py collectl/${model} -r -o fig/${model}
done
