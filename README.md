# llama3.cuda

`llama3.cuda` is a pure C/CUDA implementation for Llama 3 model.

Following on from my last implementation of the [Llama 3 model in pure NumPy](https://github.com/likejazz/llama3.np), this time I implemented the [Llama 3 model in pure C/CUDA (This repository!)](https://github.com/likejazz/llama3.cuda).

The Llama model implementation and UTF-8 tokenizer implementation were based on llama2.c previous implemented by [Andrej Karpathy](https://github.com/karpathy/llama2.c), while the CUDA code adopted the kernel implemented by [rogerallen](https://github.com/rogerallen/llama2.cu). It also heavily referenced the early CUDA kernel implemented by [ankan-ban](https://github.com/ankan-ban/llama2.cu).

The key features are:
- No dependency  
It's simple, readable, and dependency-free to make it easy to compile anywhere. Both Makefile/CMake are supported.
- No C++  
It's a pure C implementation that does not use C++, and most values are treated as pointers.
- One single file  
Even including a lot of boilerplate code, such as UTF-8 byte sequence processing, It kept the entire code to under 900 lines in a single file.
- Same result  
To achieve exactly the same results as the [NumPy implementation](https://github.com/likejazz/llama3.np), I debugged the logit values manually to reduce the floating-point arithmetic error rate, and reduced error rate to less than 0.5%.
- High performance  
When the [NumPy implementation](https://github.com/likejazz/llama3.np) on the M2 MacBook Air processed 33 tokens/s, while the CUDA version processed 2,823 tokens/s on a NVIDIA 4080 SUPER, which is about 85 times faster. This experiment really showed us why we should use GPU.

## Usage

```shell
$ make
$ ./runcuda "I have a dream"
"""
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.
One day, he was walking in the garden when he saw
Token count: 50, elapsed: 0.017000s, 2823 tokens/s
"""
```

## Next steps

I did some patches, including tokenizer, to ensure the same results as the [NumPy version](https://github.com/likejazz/llama3.np), and interestingly, I noticed that Andrej Karpathy commented in the tokenizer code "I don't have the energy to read more of the sentencepiece code to figure out what it's doing". I spent a lot of time fixing this code, but unfortunately, I didn't get good results and had to do it as a messy monkey patch. I'll try again in the future with further refinements.

In the future, I will try to verify it on other platforms besides CUDA by using AMD's ROCm implementation and Intel's oneAPI implementation. Also, there is still an issue that Multi-Head Attention is handled by a single kernel, which has a similar effect to Flash Attention, but it is somewhat inefficient because it performs GEMV operations instead of GEMM operations. In the future, I plan to improve this and implement Flash Attention correctly.

## Citing llama3.cuda

If you use or discuss `llama3.cuda` in your academic research, please cite the project to help spread awareness:

```
@misc{llama3.cuda,
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
- [llama3.np](https://github.com/likejazz/llama3.np) - @likejazz,
My previous implementation of the Llama 3 model in pure NumPy.

For more information on implement Llama 3 model, see the following article I wrote:
- [Llama 3 implemented in pure NumPy](https://docs.likejazz.com/llama3.np/)

# License
MIT
