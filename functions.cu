#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void Plot_kernel(uchar4 *ptr, unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * blockDim.x * gridDim.x;

	unsigned char* f_r, *f_g, *f_b;
	f_r = (unsigned char*)((char*)R_input + y*i_size);
	f_g = (unsigned char*)((char*)G_input + y*i_size);
	f_b = (unsigned char*)((char*)B_input + y*i_size);
	ptr[offset].x = f_r[x];
	ptr[offset].y = f_g[x];
	ptr[offset].z = f_b[x];
	ptr[offset].w = 255;
}

__device__ int log2(int N)
{
	int k = N, i = 0;
	while(k)
	{
		k >>= 1;
		i++;
	}
	return i - 1;
}

__device__ int reverse(int N, int n)
{
	int p = 0;
	for(int j = 1; j <= log2(N); j++)
	{
		if(n & (1 << (log2(N) - j)))
			p |= 1 << (j - 1);
	}
	return p;
}

__device__ void ordina_x(float *complex_r, float *complex_i,float *real_d_out, float *imagi_d_out,int row, int col, int x)
{
	int N = row, a;
	for(int i = 0; i < N; i++)
	{
		a = reverse((int)N, i);
		real_d_out[i*col + x] = complex_r[a*col + x];
		imagi_d_out[i*col + x] = complex_i[a*col + x];}
		for(int j = 0; j < N; j++)
		{
			complex_r[j*col + x] = real_d_out[j*col + x];
			complex_i[j*col + x] = imagi_d_out[j*col + x];}
		}

__device__ void ordina_y(float *complex_r, float *complex_i,float *real_d_out, float *imagi_d_out,int row, int col, int y)
{
	int N = row, a;
	for(int i = 0; i < N; i++)
	{
		a = reverse((int)N, i);
		real_d_out[y*col + i] = complex_r[y*col + a];
		imagi_d_out[y*col + i] = complex_i[y*col + a];
	}
	for(int j = 0; j < N; j++)
	{
		complex_r[y*col + j] = real_d_out[y*col + j];
		complex_i[y*col + j] = imagi_d_out[y*col + j];
	}
}

__device__ void Func_FFT_X(float *complex_r, float *complex_i,int row, int col, int x)
{
	int n = 1, N = row;
	int a = N/2;
	float temp_real, temp_imagi;
	float t_r, t_i, a_r, a_i;
	for(int j = 0; j < log2(N); j++)
	{
		for (int i = 0; i < N; i++)
		{
			if(!(i & n))
			{
				temp_real = complex_r[x + (i * col)];
				temp_imagi = complex_i[x + (i * col)];
				a_r = cos((-2) * ((i * a) % (n * a)) * PI / N);
				a_i = sin((-2) * ((i * a) % (n * a)) * PI / N);
				t_r = (a_r*complex_r[x + (i + n)*col]) - (a_i*complex_i[x + (i + n)*col]);
				t_i = (a_i*complex_r[x + (i + n)*col]) + (a_r*complex_i[x + (i + n)*col]);
				complex_r[x + (i * col)] += t_r;
				complex_i[x + (i * col)] += t_i;
				complex_r[x + (i + n)*col] = temp_real - t_r;
				complex_i[x + (i + n)*col] = temp_imagi - t_i;
			}
		}
		n *= 2;
		a = a/2;
	}
}

__device__ void Func_FFT_Y(float *complex_r, float *complex_i,int row, int col, int y)
{
	int n = 1, N = col;
	int a = N/2;
	float temp_real, temp_imagi;
	float t_r, t_i, a_r, a_i;
	for(int j = 0; j < log2(N); j++)
	{
		for (int i = 0; i < N; i++)
		{
			if(!(i & n))
			{
				temp_real = complex_r[i + (y * col)];
				temp_imagi = complex_i[i + (y * col)];
				a_r = cos(-2 * ((i * a) % (n * a)) * PI/ N);
				a_i = sin(-2 * ((i * a) % (n * a)) * PI/ N);
				t_r = (a_r*complex_r[(i + n) + y*col]) - (a_i*complex_i[(i + n) + y*col]);
				t_i = (a_i*complex_r[(i + n) + y*col]) + (a_r*complex_i[(i + n) + y*col]);
				complex_r[i + (y * col)] += t_r;
				complex_i[i + (y * col)] += t_i;
				complex_r[(i + n) + y*col] = temp_real - t_r;
				complex_i[(i + n) + y*col] = temp_imagi - t_i;
			}
		}
		n *= 2;
		a = a/2;
	}
}

__global__ void FFT_X(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,float *complex_r, float *complex_i,float *real_d_out, float *imagi_d_out,unsigned char *r_dataC, unsigned char *g_dataC,unsigned char *b_dataC, unsigned long col, unsigned long row,unsigned long colF, unsigned long rowF ) 
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	float temp;
	if(x < col)
	{
		for (int i = 0; i < row; i++)
		{
			complex_r[x + (i * colF)] = 0.2989 * R_input[x + (i * i_size)] +  0.587 * G_input[x + (i * i_size)] + 0.1140 * B_input[x + (i * i_size)];
			complex_i[x + (i * colF)] = 0;
		}
		for (int i = row; i < rowF; i++)
		{
			complex_r[x + (i * colF)] = 0;
			complex_i[x + (i * colF)] = 0;
		}
	}
	else
	{
		for (int i = 0; i < rowF; i++) 
		{
			complex_r[x + (i * colF)] = 0;
			complex_i[x + (i * colF)] = 0;
		}
	}
	ordina_x(complex_r, complex_i, real_d_out, imagi_d_out, rowF, colF, x);
	Func_FFT_X(complex_r, complex_i, rowF, colF, x);
	for (int i = 0; i < rowF/2; i++)
	{
		temp = complex_r[x + (i * colF)];
		complex_r[x + (i * colF)] = complex_r[x + ((i + rowF/2) * colF)];
		complex_r[x + ((i + rowF/2) * colF)] = temp;
		temp = complex_i[x + (i * colF)];
		complex_i[x + (i * colF)] = complex_i[x + ((i + rowF/2) * colF)];
		complex_i[x + ((i + rowF/2) * colF)] = temp;
	}
}

__global__ void FFT_Y(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,float *complex_r, float *complex_i,float *real_d_out, float *imagi_d_out,unsigned char *r_dataC, unsigned char *g_dataC,unsigned char *b_dataC, unsigned long col, unsigned long row,unsigned long colF, unsigned long rowF )
{
	int y = threadIdx.x + (blockIdx.x * blockDim.x);
	float temp;
	ordina_y(complex_r, complex_i, real_d_out, imagi_d_out, rowF, colF, y);
	Func_FFT_Y(complex_r, complex_i, rowF, colF, y);
	for (int i = 0; i < colF/2; i++)
	{
		temp = complex_r[i + (y * colF)];
		complex_r[i + (y * colF)] = complex_r[(i + colF/2) + (y * colF)];
		complex_r[(i + colF/2) + (y * colF)] = temp;
		temp = complex_i[i + (y * colF)];
		complex_i[i + (y * colF)] = complex_i[(i + colF/2) + (y * colF)];
		complex_i[(i + colF/2) + (y * colF)] = temp;
	}

	unsigned char v;
	int a = (colF/2) - (col/2);
	int temp_b = (rowF/2) - (row/2);
	if( y >= temp_b)
		for (int i = a; i < (colF/2) + (col/2); i++)
		{
			v = (unsigned char)(20*log10(sqrt((complex_r[i + (y * colF)]*complex_r[i + (y * colF)]) + (complex_i[i + (y * colF)]*complex_i[i + (y * colF)]))));
			r_dataC[(i - a ) + (y - temp_b) * i_size] = v;
			g_dataC[(i - a) + (y - temp_b) * i_size] = v;
			b_dataC[(i - a) + (y - temp_b) * i_size] = v;
		}
}

__global__ void erosion(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned char *r_dataC, unsigned char *g_dataC,unsigned char *b_dataC, unsigned long col, unsigned long row,float *mask, unsigned int dim, int m)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	int offset2, ximg, yimg;
	int c = 0;
	unsigned char color;
	int end = dim/2, ini = -end, k = 0;

	for (int i = ini; i <= end; i++)
	{
		ximg = x + i;
		for (int j = ini; j <= end; j++)
		{
			yimg = y + j;
			offset2 = ximg + yimg * i_size;
			if (ximg < col && yimg < row)
				if (ximg > 0 && yimg > 0)
					c += (R_input[offset2]*mask[k]);
			k++;
		}
	}
	if(c < m) color = 0;
	else color = 255;
	r_dataC[offset] = color;
	g_dataC[offset] = color;
	b_dataC[offset] = color;
}

__global__ void histogramGray(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned int *hist) 
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	R_input[offset] = 0.2989 * R_input[offset] +  0.587 * G_input[offset] + 0.1140 * B_input[offset];
	G_input[offset] = R_input[offset];
	B_input[offset] = R_input[offset];
	atomicAdd( &(hist[R_input[offset]]), 1);
}

__global__ void binary(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,int um)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	unsigned char c;
	if (R_input[offset] > um) c = 255;
	else c = 0;
	R_input[offset] = c;
	G_input[offset] = c;
	B_input[offset] = c;
}

__global__ void Copy(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned char *R_output, unsigned char *G_output,unsigned char *B_output)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	R_output[offset] = R_input[offset];
	G_output[offset] = G_input[offset];
	B_output[offset] = B_input[offset];
}

__global__ void median_filter(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned char *r_dataC, unsigned char *g_dataC,unsigned char *b_dataC, unsigned long col, unsigned long row,unsigned int dim)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	int offset2, ximg, yimg;
	unsigned char temp_r = 0, temp_g = 0, temp_b = 0, temp;
	int end = dim/2, ini = -end, k = 0, n = 0, i, j;
	int hr[9];
	int hg[9];
	int hb[9];

	for (i = ini; i <= end; i++)
	{
		ximg = x + i;
		for (j = ini; j <= end; j++)
		{
			yimg = y + j;
			offset2 = ximg + yimg * i_size;
			if (ximg < col && yimg < row)
				if (ximg > 0 && yimg > 0)
				{
					hr[n] = R_input[offset2];
					hg[n] = G_input[offset2];
					hb[n] = B_input[offset2];
					n++;
				}
			k++;
		}
	}
	for (i = 0; i < n; i++)
		for (j= i + 1; j < n; j++)
			if (hr[j] < hr[i])
			{
				temp = hr[j];
				hr[j] = hr[i];
				hr[i] = temp;
			}
	for (i = 0; i < n; i++)
		for (j= i + 1; j < n; j++)
			if (hg[j] < hg[i])
			{
				temp = hg[j];
				hg[j] = hg[i];
				hg[i] = temp;
			}

	for (i = 0; i < n; i++)
		for (j= i + 1; j < n; j++)
			if (hb[j] < hb[i])
			{
				temp = hb[j];
				hb[j] = hb[i];
				hb[i] = temp;
			}

	if(n%2 == 1)
	{
		temp_r = hr[(n/2)];
		temp_g = hg[(n/2)];
		temp_b = hb[(n/2)];
	}
	else
	{
		temp_r = hr[(n/2)] + hr[(n/2) - 1];
		temp_g = hg[(n/2)] + hg[(n/2) - 1];
		temp_b = hb[(n/2)] + hb[(n/2) - 1];
	}

	r_dataC[offset] = temp_r;
	g_dataC[offset] = temp_g;
	b_dataC[offset] = temp_b;
}
__global__ void Operador_Convolucion(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned char *r_dataC, unsigned char *g_dataC,unsigned char *b_dataC, unsigned long col, unsigned long row,float *mask, unsigned int dim)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	int offset2, ximg, yimg;
	unsigned char temp_r = 0, temp_g = 0, temp_b = 0;
	int end = dim/2, ini = -end, k = 0;

	for (int i = ini; i <= end; i++)
	{
		ximg = x + i;
		for (int j = ini; j <= end; j++)
		{
			yimg = y + j;
			offset2 = ximg + yimg * i_size;
			if (ximg < col && yimg < row)
				if (ximg > 0 && yimg > 0)
				{
					temp_r += R_input[offset2]*mask[k];
					temp_g += G_input[offset2]*mask[k];
					temp_b += B_input[offset2]*mask[k];
				}
			k++;
		}
	}
	r_dataC[offset] = temp_r;
	g_dataC[offset] = temp_g;
	b_dataC[offset] = temp_b;
}

__global__ void Get_Histogram(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned int *hist_r,unsigned int *hist_g,unsigned int *hist_b)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	atomicAdd( &(hist_r[R_input[offset]]), 1);
	atomicAdd( &(hist_g[G_input[offset]]), 1);
	atomicAdd( &(hist_b[B_input[offset]]), 1);
}

__global__ void Equalization_GPU(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size,unsigned char *r_dataE, unsigned char *g_dataE,unsigned char *b_dataE,unsigned int *hist_r,unsigned int *hist_g,unsigned int *hist_b)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	r_dataE[offset] = hist_r[R_input[offset]];
	g_dataE[offset] = hist_g[G_input[offset]];
	b_dataE[offset] = hist_b[B_input[offset]];
}

__global__ void Rotation_op(uchar4 *ptr, unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size, float a,unsigned long col, unsigned long row)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * blockDim.x * gridDim.x;
	x = x - (blockDim.x * gridDim.x / 2);
	y = y - (blockDim.y * gridDim.y / 2);

	unsigned char* f_r, *f_g, *f_b;

	int ximg = (x*cos(a) + y*sin(a)) + (col/2), yimg = (y*cos(a) - x*sin(a)) + (row/2);
	if (ximg < col && yimg < row)
	{
		f_r = (unsigned char*)((char*)R_input + yimg*i_size);
		f_g = (unsigned char*)((char*)G_input + yimg*i_size);
		f_b = (unsigned char*)((char*)B_input + yimg*i_size);
		ptr[offset].x = f_r[ximg];
		ptr[offset].y = f_g[ximg];
		ptr[offset].z = f_b[ximg];
		ptr[offset].w = 255;
	}
	else
	{
		ptr[offset].x = 0;
		ptr[offset].y = 0;
		ptr[offset].z = 0;
		ptr[offset].w = 255;
	}
}

__global__ void Scaling_op(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input,unsigned char *R_output, unsigned char *G_output,unsigned char *B_output,size_t i_size, size_t pitch2, float s,unsigned long col, unsigned long row)
{
	float x = threadIdx.x + (blockIdx.x * blockDim.x);
	float y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * pitch2;
	x = x - (DIM / 2);
	y = y - (DIM / 2);
	unsigned char* f_r, *f_g, *f_b;
	x /= s; y /= s;

	int ximg = x + (col/2), yimg = y + (row/2);
	if (ximg < (col - 1) && yimg < (row - 1))
	{
		f_r = (unsigned char*)((char*)R_input + yimg*i_size);
		f_g = (unsigned char*)((char*)G_input + yimg*i_size);
		f_b = (unsigned char*)((char*)B_input + yimg*i_size);
		float cx = x - floor(x);
		float cy = y - floor(y);
		float R1 = f_r[ximg]*(1 - cx) + f_r[ximg + 1]*(cx);
		float R2 = f_r[ximg + i_size]*(1 - cx) + f_r[ximg + 1 + i_size]*(cx);
		R_output[offset] = R1*(1 - cy) + R2*(cy);
		R1 = f_g[ximg]*(1 - cx) + f_g[ximg + 1]*(cx);
		R2 = f_g[ximg + i_size]*(1 - cx) + f_g[ximg + 1 + i_size]*(cx);
		G_output[offset] = R1*(1 - cy) + R2*(cy);
		R1 = f_b[ximg]*(1 - cx) + f_b[ximg + 1]*(cx);
		R2 = f_b[ximg + i_size]*(1 - cx) + f_b[ximg + 1 + i_size]*(cx);
		B_output[offset] = R1*(1 - cy) + R2*(cy);
	}
	else
	{
		R_output[offset] = 0;
		G_output[offset] = 0;
		B_output[offset] = 0;
	}
}

__global__ void PPnoise(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input, size_t i_size, int noiseP, int seed)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = x + y * i_size;
	curandState_t state;
	curand_init(seed, x,  y, &state);
	unsigned char noise = (unsigned char)(curand(&state) % 100);
	if(curand(&state) % 100 < noiseP)
	{
		noise = 255 * (noise % 2);
		R_input[offset] = noise;
		G_input[offset] = noise;
		B_input[offset] = noise;
	}
}
