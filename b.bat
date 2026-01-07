@echo off
clang demo.c lib\ml.c -gfull -o d.exe -fsanitize=address
@echo on
