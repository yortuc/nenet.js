# Nenet
> vectorized neural network library for browser and node

## Easy to use

```js
const x = [[2, 4, 6], 
           [3, 6, 9]];

const y = [[0.25, 0.5, 0.7], 
           [0.5, 0.7, 0.9]];

const nn = new Nenet([
		["input", x],
		["hidden", 6],
		["hidden", 6],
		["hidden", 6],
		["output", y]
	])
	.train(100);

console.log(nn.y_pred);
```

## Examples

Please check the `examples` folder

1. iris classification

2. mnist classification

3. house price regression

4. boolean operations XOR, NXOR, OR, AND

## Roadmap

Seperate neural network and trainer

Implement online and mini-batch training 

Ipmlement momentum

Lab
- Web app (redux, react, d3, nenet)
- Visualize neural network and workflow.
- Create, run 
- Plot error and weights.
- Visualize propagated errors