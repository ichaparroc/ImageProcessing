//nvcc main.cu -lglut -lGLEW -lGL -lm -ccbin clang-3.8 -lstdc++

#include <assert.h>
#include <cstdio>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda_gl_interop.h>
#include "functions.cu"
#include <string.h>

using namespace std;

#define WIDTH  1280
#define HEIGHT 960
#define DIM 1600
#define PI 3.14159265

static int sub_00;
static int sub_01;
static int sub_02;
static int sub_03;
static int sub_04;

bool Flag_Equl = 0, Flag_Filt = 0, Flag_01 = 0, Flag_Med = 0, Flag_PPnoise = 0;
bool Flag_Reset = 0, Flag_Ero = 0, Flag_Gray = 0, Flag_BW = 0, Flag_Fourier = 0;
long long int sizeImage;
float Scale_Factor;
float Rotation_Factor;
unsigned long widht, height;
int Num_Cols, Num_Rows, Dim_Con, Num_Rows_Fourier, Num_Cols_Fourier, Max_E;
size_t or_size, mor_size, equ_size, fou_size;
unsigned char *Image_R, *Image_G, *Image_B;
unsigned char *Image_R_bk, *Image_G_bk, *Image_B_bk;
unsigned char *Equalizar_R, *Equalizar_G, *Equalizar_B;
unsigned char *Convol_R, *Convol_G, *Convol_B;
unsigned char *Fourier_R, *Fourier_G, *Fourier_B;
unsigned char *Morfo_R, *Morfo_G, *Morfo_B;
float *Val_Real, *Val_Real_out, *Val_Imag, *Val_Imag_out;
unsigned int *d_his_r;
unsigned int *d_his_g;
unsigned int *d_his_b;
float *DMask;
float *Mask = (float*)malloc(625*sizeof(float));

int Threshold(unsigned char *r_data, unsigned char *g_data, unsigned char *b_data, size_t pitch);

void Equalization_PC (unsigned char *r_data,unsigned char *g_data,unsigned char *b_data, size_t pitch,unsigned char *r_dataE,unsigned char *g_dataE,unsigned char *b_dataE );

void FFT();

typedef struct BMP_Info{
	unsigned long bytesInHeader;
	unsigned long widht;
	unsigned long height;
	unsigned int planes;
	unsigned int bitsPerPixel;
	unsigned long compression;
	unsigned long sizeImage;
	unsigned long hResolution;
	unsigned long vResolution;
	unsigned long nIndexes;
	unsigned long nIIndexes;
	char type[3];
	unsigned long size;
	char reserved[5];
	unsigned long offset;
} BMP_Info;

unsigned long Turn_Data_Long(FILE* fp)
{
	uint32_t data32;
	fread (&(data32),4, 1,fp);
	unsigned long data = (unsigned long)data32;
	return data;
}

unsigned int Turn_Data_Int(FILE* fp)
{
	uint16_t data16;
	fread (&(data16), 2, 1, fp);
	unsigned int data = (unsigned int)data16;
	return data;
}

void Read_Image(FILE* fp, BMP_Info* Image_Raw)
{
	fgets(Image_Raw->type, 3, fp);
	Image_Raw->size = Turn_Data_Long(fp);
	fgets(Image_Raw->reserved, 5, fp);
	Image_Raw->offset = Turn_Data_Long(fp);
	Image_Raw->bytesInHeader = Turn_Data_Long(fp);
	Image_Raw->widht = Turn_Data_Long(fp);
	Image_Raw->height = Turn_Data_Long(fp);
	Image_Raw->planes = Turn_Data_Int(fp);
	Image_Raw->bitsPerPixel = Turn_Data_Int(fp);
	Image_Raw->compression = Turn_Data_Long(fp);
	Image_Raw->sizeImage = Turn_Data_Long(fp);
	Image_Raw->hResolution = Turn_Data_Long(fp);
	Image_Raw->vResolution = Turn_Data_Long(fp);
	Image_Raw->nIndexes = Turn_Data_Long(fp);
	Image_Raw->nIIndexes = Turn_Data_Long(fp);
}

FILE *fp;
BMP_Info Image_Raw;

void display()
{
	GLuint bufferObj;
	struct cudaGraphicsResource* resource;
	bool Flag_02 = 1;
	glClearColor( 0.0, 0.0, 0.0, 1.0  );
	glClear( GL_COLOR_BUFFER_BIT );

	glGenBuffers( 1, &bufferObj );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, widht * height * 4, NULL, GL_DYNAMIC_DRAW_ARB );
	cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );
	uchar4* devPtr;
	size_t size;
	cudaGraphicsMapResources( 1, &resource, NULL ) ;
	cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource );

	dim3 grids(Num_Cols/16,Num_Rows/16);
	dim3 threads(16, 16);
	dim3 grids_01(DIM/16,DIM/16);
	dim3 threads_01(16, 16);
	dim3 grids_02(widht/16,height/16);
	dim3 threads_02(16, 16);

	if(Flag_Reset)
	{
		Copy<<<grids,threads>>>(Image_R_bk, Image_G_bk, Image_B_bk, or_size,
			Image_R, Image_G, Image_B);
		Flag_Reset = 0;
	}

	if(Flag_Ero)
	{
		erosion<<<grids,threads>>>(Image_R, Image_G, Image_B,
			or_size, Image_R, Image_G, Image_B, Num_Cols, Num_Rows, DMask, Dim_Con, Max_E);
		Flag_Ero = 0;
	}

	if(Flag_Gray)
	{
		histogramGray<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size, d_his_r);
		Flag_Gray = 0;
	}

	if(Flag_BW)
	{
		Threshold (Image_R, Image_G, Image_B, or_size );
		Flag_BW = 0;
	}

	if (Flag_PPnoise)
	{
		PPnoise<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size, 1, rand()%100);
		Flag_PPnoise = 0;
	}

	if(Flag_Fourier)
	{
		FFT();
		Flag_Fourier = 0;
	}

	if (Flag_Med) 
	{
		median_filter<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size,
			Convol_R, Convol_G, Convol_B, Num_Cols, Num_Rows, 3);
		Flag_02 = 0;
	}

	if (Flag_01)
	{
		if (Flag_Filt)
		{
			Equalization_PC (Image_R, Image_G, Image_B, or_size,
				Equalizar_R, Equalizar_G, Equalizar_B );
			Operador_Convolucion<<<grids,threads>>>(Equalizar_R, Equalizar_G, Equalizar_B,
				or_size, Convol_R, Convol_G, Convol_B, Num_Cols, Num_Rows, DMask, Dim_Con);
		}
		else
		{
			Equalization_PC (Image_R, Image_G, Image_B, or_size,
				Convol_R, Convol_G, Convol_B );
		}
		Flag_02 = 0;
	}
	else if (Flag_Filt)
	{
		Operador_Convolucion<<<grids,threads>>>(Image_R, Image_G, Image_B,
			or_size, Convol_R, Convol_G, Convol_B, Num_Cols, Num_Rows, DMask, Dim_Con);
		if (Flag_Equl)
			Equalization_PC (Convol_R, Convol_G, Convol_B, or_size,
				Convol_R, Convol_G, Convol_B );
		Flag_02 = 0;
	}

	if (Flag_02)
	{
		Scaling_op<<<grids_01,threads_01>>>(Image_R, Image_G, Image_B, Morfo_R, Morfo_G, Morfo_B,
			or_size, mor_size, Scale_Factor, Num_Cols, Num_Rows);
	}
	else
	{
		Scaling_op<<<grids_01,threads_01>>>(Convol_R, Convol_G, Convol_B, Morfo_R, Morfo_G, Morfo_B,
			or_size, mor_size, Scale_Factor, Num_Cols, Num_Rows);
	}

	Rotation_op<<<grids_02,threads_02>>>( devPtr, Morfo_R, Morfo_G, Morfo_B,
		mor_size, Rotation_Factor, DIM, DIM);

	cudaGraphicsUnmapResources( 1, &resource, NULL ) ;
	glDrawPixels( widht, height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glutSwapBuffers();
	cudaGraphicsUnregisterResource( resource ) ;
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	glDeleteBuffers( 1, &bufferObj );
}

static void key_func( unsigned char key, int x, int y )
{
	switch (key)
	{
		case 's':
		if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) + 0.1);
		else Scale_Factor -= 0.1;
		break;
		case 'w':
		if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) - 0.1);
		else Scale_Factor += 0.1;
		break;
		case 'd':
		Rotation_Factor -= 0.05*PI;
		break;
		case 'a':
		Rotation_Factor += 0.05*PI;
		break;
	}

	display();
}

int Threshold(unsigned char *r_data, unsigned char *g_data, unsigned char *b_data, size_t pitch)
{
	unsigned int his_size = sizeof(unsigned int)*256;
	unsigned int *his = (unsigned int*)malloc(his_size);

	cudaMemset( d_his_r, 0, his_size);
	dim3 grids(Num_Cols,Num_Rows);
	dim3 threads(1, 1);
	histogramGray<<<grids,threads>>>(r_data, g_data, b_data, pitch, d_his_r);
	cudaMemcpy(his, d_his_r, his_size, cudaMemcpyDeviceToHost);
	int m = Num_Cols*Num_Rows/2, h = 0, um, i;
	for (i = 0; i < 256; i++)
	{
		h += his[i];
		if (h > m)
		{
			um = i;
			break;
		}
	}
	binary<<<grids,threads>>>(r_data, g_data, b_data, pitch, um);
	return um;
}

void FFT()
{
	FFT_X<<<Num_Cols_Fourier/128, 128>>>(Image_R, Image_G, Image_B,or_size, Val_Real, Val_Imag, Val_Real_out, Val_Imag_out,Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Num_Cols_Fourier, Num_Rows_Fourier);
	FFT_Y<<<Num_Rows_Fourier/128, 128>>>(Image_R, Image_G, Image_B,or_size, Val_Real, Val_Imag, Val_Real_out, Val_Imag_out,Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Num_Cols_Fourier, Num_Rows_Fourier);
}

void Equalization_PC (unsigned char *r_data, unsigned char *g_data,unsigned char *b_data, size_t pitch,unsigned char *r_dataE, unsigned char *g_dataE,unsigned char *b_dataE )
{
	int i;
	unsigned int his_size = sizeof(unsigned int)*256;
	float hisAc_size = sizeof(float)*256;

	unsigned int *his_r = (unsigned int*)malloc(his_size);
	unsigned int *his_g = (unsigned int*)malloc(his_size);
	unsigned int *his_b = (unsigned int*)malloc(his_size);

	float *hisAc_r = (float*)malloc(hisAc_size);
	float *hisAc_g = (float*)malloc(hisAc_size);
	float *hisAc_b = (float*)malloc(hisAc_size);

	cudaMemset( d_his_r, 0, his_size);
	cudaMemset( d_his_g, 0, his_size);
	cudaMemset( d_his_b, 0, his_size);

	dim3 grids(Num_Cols,Num_Rows);
	dim3 threads(1, 1);
	Get_Histogram<<<grids,threads>>>(r_data, g_data, b_data, pitch, d_his_r, d_his_g, d_his_b);

	cudaMemcpy(his_r, d_his_r, his_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(his_g, d_his_g, his_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(his_b, d_his_b, his_size, cudaMemcpyDeviceToHost);

	hisAc_r[0] = ((float)his_r[0])/sizeImage;
	hisAc_g[0] = ((float)his_g[0])/sizeImage;
	hisAc_b[0] = ((float)his_b[0])/sizeImage;
	for (i = 1; i < 256; i++)
	{
		hisAc_r[i] = hisAc_r[i-1] + (((float)his_r[i])/sizeImage);
		hisAc_g[i] = hisAc_g[i-1] + (((float)his_g[i])/sizeImage);
		hisAc_b[i] = hisAc_b[i-1] + (((float)his_b[i])/sizeImage);
	}
	his_r[0] = 0;
	his_g[0] = 0;
	his_b[0] = 0;

	for (i = 1; i < 255; i++)
	{
		his_r[i] = (unsigned int)(hisAc_r[i - 1]*255);
		his_g[i] = (unsigned int)(hisAc_g[i - 1]*255);
		his_b[i] = (unsigned int)(hisAc_b[i - 1]*255);
	}
	his_r[255] = 255;
	his_g[255] = 255;
	his_b[255] = 255;

	cudaMemcpy(d_his_r, his_r, his_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_his_g, his_g, his_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_his_b, his_b, his_size, cudaMemcpyHostToDevice);

	Equalization_GPU<<<grids,threads>>>(r_data, g_data, b_data,
		or_size, r_dataE, g_dataE, b_dataE, d_his_r, d_his_g, d_his_b);
}

void call_back_function(int val)
{
	switch (val)
	{
		case 3:
			if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) + 0.15);
			else Scale_Factor -= 0.15;
			break;
		case 2:
			if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) - 0.15);
			else Scale_Factor += 0.15;
			break;
		case 4:
			Rotation_Factor -= 0.01*PI;
			break;
		case 5:
			Rotation_Factor += 0.01*PI;
			break;
		case 6:
			Flag_Equl = 1;
			if (Flag_Filt)
			{
				Flag_01 = 0;
			}
			else
			{
				Flag_01 = 1;
			}
			break;
		case 7:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = 1/9.0; Mask[1] = 1/9.0; Mask[2] = 1/9.0;
			Mask[3] = 1/9.0; Mask[4] = 1/9.0; Mask[5] = 1/9.0;
			Mask[6] = 1/9.0; Mask[7] = 1/9.0; Mask[8] = 1/9.0;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 8:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = 1/16.0; Mask[1] = 2/16.0; Mask[2] = 1/16.0;
			Mask[3] = 2/16.0; Mask[4] = 4/16.0; Mask[5] = 2/16.0;
			Mask[6] = 1/16.0; Mask[7] = 2/16.0; Mask[8] = 1/16.0;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 9:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = 0; Mask[1] = -1; Mask[2] = 0;
			Mask[3] = -1; Mask[4] = 5; Mask[5] = -1;
			Mask[6] = 0; Mask[7] = -1; Mask[8] = 0;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 10:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = -1; Mask[1] = -1; Mask[2] = -1;
			Mask[3] = -1; Mask[4] = 8; Mask[5] = -1;
			Mask[6] = -1; Mask[7] = -1; Mask[8] = -1;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 11:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = -1; Mask[1] = 0; Mask[2] = 1;
			Mask[3] = -1; Mask[4] = 0; Mask[5] = 1;
			Mask[6] = -1; Mask[7] = 0; Mask[8] = 1;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
		case 12:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = -3; Mask[1] = 0; Mask[2] = 3;
			Mask[3] = -10; Mask[4] = 0; Mask[5] = 10;
			Mask[6] = -3; Mask[7] = 0; Mask[8] = 3;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 13:
			Flag_Filt = 1;
			Dim_Con = 3;
			Mask[0] = 3; Mask[1] = 10; Mask[2] = 3;
			Mask[3] = 0; Mask[4] = 0; Mask[5] = 0;
			Mask[6] = -3; Mask[7] = -10; Mask[8] = -3;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
		case 14:
			Flag_PPnoise = 1;
			break;
		case 15:
			Flag_Med = 1;
			break;
		case 16:
			Flag_Fourier = 1;
			break;
		case 17:
			Flag_Reset = 1;
			Scale_Factor = 1;
			Rotation_Factor = 0;
			break;
		case 18:
			Flag_Ero = 1;
			Dim_Con = 3;
			Mask[0] = 0; Mask[1] = 1; Mask[2] = 0;
			Mask[3] = 1; Mask[4] = 1; Mask[5] = 1;
			Mask[6] = 0; Mask[7] = 1; Mask[8] = 0;
			Max_E = 255;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 19:
			Flag_Ero = 1;
			Dim_Con = 3;
			Mask[0] = 0; Mask[1] = 1; Mask[2] = 0;
			Mask[3] = 1; Mask[4] = 1; Mask[5] = 1;
			Mask[6] = 0; Mask[7] = 1; Mask[8] = 0;
			Max_E = 5* 255;
			cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 20:
			Flag_Gray = 1;
			break;
		case 21:
			Flag_BW = 1;
			break;
		case 1:
			exit(0);
			break;
		default:
			{
			}
	}
	display();
}

void Create_call_back_function(void)
{
	sub_00 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Ampliar", 2);
	glutAddMenuEntry("Reducir", 3);

	sub_01 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Derecha", 4);
	glutAddMenuEntry("Izquierda", 5);

	sub_02 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Media", 7);
	glutAddMenuEntry("Gaussian blur", 8);
	glutAddMenuEntry("Sharpen", 9);
	glutAddMenuEntry("Edge Detection", 10);
	glutAddMenuEntry("Laplaciano", 11);
	glutAddMenuEntry("Sobel X", 12);
	glutAddMenuEntry("Sobel Y", 13);

	sub_03 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Erosion", 18);
	glutAddMenuEntry("Dilatacion", 19);

	sub_04 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Grises", 20);
	glutAddMenuEntry("Binario", 21);

	glutCreateMenu(call_back_function);
	glutAddSubMenu("Escalamiento", sub_00);
	glutAddSubMenu("Rotacion", sub_01);
	glutAddMenuEntry("Ecualizacion", 6);
	glutAddSubMenu("Convoluciones", sub_02);
	glutAddMenuEntry("Filtro de la Mediana", 15);
	glutAddSubMenu("Op. Morfologicas", sub_03);
	glutAddSubMenu("Escalas", sub_04);
	glutAddMenuEntry("Agregar Ruido", 14);
	glutAddMenuEntry("Fourier", 16);
	glutAddMenuEntry("Original", 17);
	glutAddMenuEntry("Salir", 1);

	glutAttachMenu(GLUT_LEFT_BUTTON);
}

int main(int argc, char** argv)
{
	time_t t;
	srand((unsigned) time(&t));
	int i = 0, j;
	unsigned char c;
	Scale_Factor = 1;
	Rotation_Factor = 0;

	const char * T;
	T = "lena.bmp";
	fp = fopen(T,"r");
	Read_Image(fp, &Image_Raw);
	i = 54;
	while(i < Image_Raw.offset)
	{
		c = fgetc(fp);
		if(feof(fp))
			break;
		i++;
	}

	Num_Cols = Image_Raw.widht, Num_Rows = Image_Raw.height;
	sizeImage = Num_Cols*Num_Rows;
	unsigned char Matrix_R[Num_Rows][Num_Cols], Matrix_G[Num_Rows][Num_Cols], Matrix_B[Num_Rows][Num_Cols];
	for ( i = 0; i < Num_Rows; i++)
	{
		for( j = 0; j < Num_Cols; j++)
		{
			if (Image_Raw.bitsPerPixel > 8) 
			{
				Matrix_B[i][j] = fgetc(fp);
				Matrix_G[i][j] = fgetc(fp);
				Matrix_R[i][j] = fgetc(fp);
				if(Image_Raw.bitsPerPixel > 24)
						c = fgetc(fp);
			}
			else if(Image_Raw.bitsPerPixel == 8)
			{
				c = getc(fp);
				Matrix_B[i][j] = c;
				Matrix_G[i][j] = c;
				Matrix_R[i][j] = c;
			}
		}
	}
	fclose(fp);

	widht = WIDTH; height = HEIGHT;

	Num_Rows_Fourier = pow(2,(int)(log(Num_Rows - 1)/log(2)) + 1);
	Num_Cols_Fourier = pow(2,(int)(log(Num_Cols - 1)/log(2)) + 1);
	unsigned int his_size = sizeof(unsigned int)*256;
	unsigned int comp_size = sizeof(float)*Num_Rows_Fourier*Num_Cols_Fourier;

	cudaMallocManaged(&d_his_r, his_size);
	cudaMallocManaged(&d_his_g, his_size);
	cudaMallocManaged(&d_his_b, his_size);

	cudaMallocManaged(&Val_Real, comp_size);
	cudaMallocManaged(&Val_Real_out, comp_size);
	cudaMallocManaged(&Val_Imag, comp_size);
	cudaMallocManaged(&Val_Imag_out, comp_size);

	cudaMallocManaged(&DMask, sizeof(float)*625);

	// Original
	cudaMallocPitch((void**)&Image_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Image_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Image_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	// Back - up
	cudaMallocPitch((void**)&Image_R_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Image_G_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Image_B_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	// Fourier
	cudaMallocPitch((void**)&Fourier_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Fourier_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Fourier_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Val_Real, &fou_size, sizeof(float)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Val_Imag, &fou_size, sizeof(float)*Num_Cols, Num_Rows);
	// Convoluciones
	cudaMallocPitch((void**)&Convol_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Convol_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Convol_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	// Equalizaciones
	cudaMallocPitch((void**)&Equalizar_R, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Equalizar_G, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	cudaMallocPitch((void**)&Equalizar_B, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
	// Rotacion y escalamiento
	cudaMallocPitch((void**)&Morfo_R, &mor_size, sizeof(unsigned char)*DIM, DIM);
	cudaMallocPitch((void**)&Morfo_G, &mor_size, sizeof(unsigned char)*DIM, DIM);
	cudaMallocPitch((void**)&Morfo_B, &mor_size, sizeof(unsigned char)*DIM, DIM);

	// Copiar al GPU
	cudaMemcpy2D(Image_R, or_size, Matrix_R, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(Image_G, or_size, Matrix_G, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(Image_B, or_size, Matrix_B, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(Image_R_bk, or_size, Matrix_R, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(Image_G_bk, or_size, Matrix_G, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);
	cudaMemcpy2D(Image_B_bk, or_size, Matrix_B, sizeof(unsigned char)*Num_Cols,sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

	glutInitWindowSize(widht, height);
	glutInit(&argc, argv);
	glutInitContextFlags(GLUT_DEBUG);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL | GLUT_DOUBLE);
	glutCreateWindow("Editor de Imagenes");

	glewInit();

	if (GLEW_KHR_debug)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	}
	else
	{
		printf("No GLEW_KHR_debug!");
	}

	Create_call_back_function();
	glutKeyboardFunc( key_func );
	glutDisplayFunc(display);
	glutMainLoop();

	cudaFree(Image_R), cudaFree(Image_G), cudaFree(Image_B);
	cudaFree(Morfo_R), cudaFree(Morfo_G), cudaFree(Morfo_B);
	cudaFree(Fourier_R), cudaFree(Fourier_G), cudaFree(Fourier_B);
	cudaFree(Convol_R), cudaFree(Convol_G), cudaFree(Convol_B);
	cudaFree(Val_Real), cudaFree(Val_Real);
	return 0;
}
