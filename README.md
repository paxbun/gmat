# Gmat

Matrix operations using CUDA



# Contents

1. [Overview](#overview)
2. [Tutorial](#tutorial)
3. [Hack Gmat](#hack-gmat)

# Overview

## C++

```c++
#include <iostream>
#include <gmat.h>

using namespace std;

int main()
{
    try
    {	
        // {{ 6, 8, 10, }}
        cout << gmat({ { 1, 2, 3 } }) + gmat({ { 5, 6, 7 } }) << endl;
        // {{ -6+0j, -4+0j, -2+0j, }}
        cout << gmatc({ { 1, 2, 3 } }) - gmatc({ { 7, 6, 5 } }) << endl;
        // {{ 0+5j, 0+12j, 0+21j, }}
        cout << gmatc({ { 1_j, 2_j, 3_j } }) * gmatc({ { 5, 6, 7 } }) << endl;
        // {{ 0.2, 0.333333, 0.428571, }}
        cout << gmat({ { 1, 2, 3 } }) / gmat({ { 5, 6, 7 } }) << endl;
        // {{ 14, 16, 18, },
        // { 31.6, 36.8, 42, },
        // { 49.2, 57.6, 66, },
        // { 66.8, 78.4, 90, }}
        cout << gmat({
            { 1, 2, 3, 4, },
            { 5, 6, 7, 8, },
            { 9, 10, 11, 12, },
            { 13, 14, 15, 16, },
        }) % gmat({
            { 1, 2, 3, },
            { 4, 5, 6, },
            { 7, 8, 9, },
            { 10, 11, 12, },
        }) / 5.0 << endl;
    }
    catch (const gmat_exception & ex)
    {
        cout << "Exception: " << ex.what() << endl;
    }
    return 0;
}
```

## C

```c
#include <stdio.h>
#include <gmat.h>

#include <stdio.h>
#include <gmat.h>

const float data[4][4] = {
	{ 1, 2, 3, 4, },
	{ 5, 6, 7, 8, },
	{ 9, 10, 11, 12, },
	{ 13, 14, 15, 16, },
};
float dres[4][4];

void print()
{
	putc('{', stdout);
	for (int i = 0; i < 4; i++)
	{
		putc('{', stdout);
		putc(' ', stdout);
		for (int j = 0; j < 4; j++)
			printf("%g, ", dres[i][j]);
		putc('}', stdout);
		if (i != 3)
			puts(",");
	}
	puts("}", stdout);
}

#define CHECK_GMAT(exp, msg) if((exp) != gmat_success) { puts(msg); rtn = 1; goto destroy; }

int main()
{
	gmat mat0, mat1, res;
	int rtn = 0;
	
	CHECK_GMAT(gmat_create(&mat0, 4, 4, 0), "Failed to initialize mat0.");
	CHECK_GMAT(gmat_create(&mat1, 4, 4, 0), "Failed to initialize mat1.");
	CHECK_GMAT(gmat_create(&res, 4, 4, 0), "Failed to initialize res.");
	
	CHECK_GMAT(gmat_copy_from(&mat0, data), "Failed to copy data.");
	CHECK_GMAT(gmat_copy_from(&mat1, data), "Failed to copy data.");
	
	CHECK_GMAT(gmat_product(&res, &mat0, &mat1), "Operation failed.");
	CHECK_GMAT(gmat_copy_to(&res, dres), "Failed to get the result.");

    // {{ 90, 100, 110, 120, },
	// { 202, 228, 254, 280, },
	// { 314, 356, 398, 440, },
	// { 426, 484, 542, 600, }}
	print();

destroy:
	gmat_destroy(&mat0);
	gmat_destroy(&mat1);
	gmat_destroy(&res);
	return 0;
}
```



# Tutorial

## System Requirements

* x86-64 based Windows PC
* CUDA-capable NVIDIA GPU
* Visual Studio 2015 or 2017

## Files

Download files from [link](https://github.com/paxbun/gmat/releases). After download, you will have 5 files:

![1](tutorials/1.png)

## Create a new project

Open visual studio, and create a new C++ Windows Console Application project by choosing **File** > **New** > **Project**. Name it **GmatPractice**.

![2](tutorials/2.png)

## Get the project ready

Right-click the project **GmatPracice** at **Solution Explorer**, and choose **Properties**. Select **C/C++** > **General** and type the path of the folder where you downloaded the files in **Additional Include Directories**. Select **Linker** > **General**, and type the same path in **Additional Library Directories**.

![3](tutorials/3.png)

Select **Linker** > **Input**, and type "gmat.lib" in **Additional Dependencies**.

![4](tutorials/4.png)

Click **OK** to save the changes.

## Manage Configuration

Since Gmat does not support x86, your first Gmat program should be based on x64. Select **x86** > **Configuration manager**.

![5](tutorials/5.png)

Select **Active Solution platform** > **x86** > **Edit...**.

![6](tutorials/6.png)

Remove **x86**.

![7](tutorials/7.png)

## Create your first Gmat program

Open **stdafx.h**, and type as follows:
```c++
#include <gmat.h>
```
Now you are ready to create your first gmat program. Open **GmatPratice.h**. What we are going to make is a program which performs multiplication of 3x2 and 2x3 matrix. The operation yields a 3x3 matrix. At the body of **main**, we create two matrix.
```c++
try {
    gmat const mat0({
        { 1, 2 },
        { 3, 4 },
        { 5, 6 }
    });
    gmat const mat1({
        { 1, 2, 3 },
        { 4, 5, 6 },
    });
```
Since Gmat has adapted **Exception handling**, you should surround your matrix operations by try-catch statement. We now calculate the product and print the result.
```c++
    std::cout << mat0 % mat1 << std::endl;
}
catch (const gmat_exception & ex)
{
    std::cout << ex.what() << std::endl;
}
```
Note that **operator\%** means matrix multiplication in the default implementation. **operator\*** is convolution operator. This is the same with **operator\/**.


# Hack Gmat

You may have noticed that there are two more files, **gmat.inl** and **gmatc.inl**. The core part of Gmat has C linkage, and the class **gmat** and **gmatc** are just wrappers. You can implement your own C++ class by editing **gmat.inl** and **gmatc.inl**. These two files are used in **gmat.h** as follows:
```c++
#include "gmat.inl"
#include "gmatc.inl"
```
