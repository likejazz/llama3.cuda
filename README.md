# llama3.cuda

`llama3.cuda` is a pure C/CUDA implementation for Llama 3 model.

---
Following on from my last implementation of the [Llama 3 model in pure NumPy](https://github.com/likejazz/llama3.np), this time I implemented the [Llama 3 model in pure C/CUDA](https://github.com/likejazz/llama3.cuda).

The Llama model implementation and UTF-8 tokenizer implementation were based on llama2.c previous implemented by [Andrej Karpathy](https://github.com/karpathy/llama2.c), while the CUDA code adopted the kernel implemented by [rogerallen](https://github.com/rogerallen/llama2.cu). In addition, I heavily referenced the early CUDA kernel implemented by [ankan-ban](https://github.com/ankan-ban/llama2.cu).

The key features are:
- I've tried to keep the code simple and readable, and It's dependency-free to make it easy to compile anywhere. Both Makefile/CMake are supported.

## Usage

```shell
$ make
$ ./runcuda "I have a dream"
"""
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.
One day, he was walking in the garden when he saw
Token count: 50, elapsed: 0.016000s, 3062 tokens/s
"""
```

## Citing llama3.cuda

If you use or discuss `llama3.cuda` in your academic research, please cite the project to help spread awareness:

```
@misc{llama3.np,
  title = {llama3.cuda: pure C/CUDA implementation for Llama 3 model},
  author = {Sang Park}, 
  howpublished = {\url{https://github.com/likejazz/llama3.cuda}},
  note = {llama3.cuda, MIT License}
  year = {2024},
}
```

# References
I've adopted most of the code from the authors below:
- [llama2.c](https://github.com/karpathy/llama2.c) - @karpathy
- [llama2.cu](https://github.com/rogerallen/llama2.cu) - @rogerallen
- [llama2.cu](https://github.com/ankan-ban/llama2.cu) - @ankan-ban

For more information on implement Llama 3 model, see the following article:
- [Llama 3 implemented in pure NumPy](https://docs.likejazz.com/llama3.np/)

# License
MIT
