#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/resource.h>
// #define SWAP(a,b)  tempr=(a);(a)=(b);(b)=tempr

// struct to hold all data up until the end of subchunk1
typedef struct {
    char chunk_id[4];
    int chunk_size;
    char format[4];
    char subchunk1_id[4];
    int subchunk1_size;
    short audio_format;
    short num_channels;
    int sample_rate;
    int byte_rate;
    short block_align;
    short bits_per_sample;
} WavHeader;

//  The four1 FFT from Numerical Recipes in C,
//  p. 507 - 508. Modified by me.
//  Note:  changed float data types to double.
//  nn must be a power of 2, and use +1 for
//  isign for an FFT, and -1 for the Inverse FFT.
//  The data is complex, so the array size must be
//  nn*2. This code assumes the array starts
//  at index 1, not 0, so subtract 1 when
//  calling the routine (see main() below).
void four1(double data[], int nn, int isign)
{
    register unsigned long n, mmax, m, j, istep, i;
    register double wtemp, wr, wpr, wpi, wi, theta;
    register double tempr, tempi;

    // register double max = -1000000.0;

    n = nn << 1;
    j = 1;

    for (i = 1; i < n; i += 2) {
	if (j > i) {
	    // SWAP(data[j], data[i]);
	    // SWAP(data[j+1], data[i+1]);
        asm("" : "=r" (data[j]), "=r" (data[i]) : "0" (data[i]), "1" (data[j]) : );
        asm("" : "=r" (data[j+1]), "=r" (data[i+1]) : "0" (data[i+1]), "1" (data[j+1]) : );
	}
	m = nn;
	while (m >= 2 && j > m) {
	    j -= m;
	    m >>= 1;
	}
	j += m;
    }

    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (6.28318530717959 / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= n; i += istep) {
            j = i + mmax;
            tempr = wr * data[j] - wi * data[j+1];
            tempi = wr * data[j+1] + wi * data[j];
            data[j] = data[i] - tempr;
            data[j+1] = data[i+1] - tempi;
            data[i] += tempr;
            data[i+1] += tempi;
            // max = abs(data[j]) > max ? abs(data[j]) : max;
            // max = abs(data[j+1]) > max ? abs(data[j+1]) : max;
            // max = abs(data[i]) > max ? abs(data[i]) : max;
            // max = abs(data[i+1]) > max ? abs(data[i+1]) : max;
            }
            wr = (wtemp = wr) * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
    // return max;
}

// Function to find the next power of 2 greater than or equal to n
int next_power_of_2(int n) {
    return pow(2, (int)(log2(n - 1) + 1));
}

void convolution(double *x, int K, double *h, double *y) {

    // Perform the DFT
    for (int k = 0, nn = 0; k < K; k++, nn += 2)
    {
	    y[nn] = ((x[nn] * h[nn]) - (x[nn+1] * h [nn+1]));
	    y[nn+1] = ((x[nn] * h[nn+1]) + (x[nn+1] * h[nn]));
	}
}

// Function to convert a short to a double in the range -1 to 1
// This is used as .wav files store data in short format (typically 16 bits, can also be extracted from the bits_per_sample header)
double shortToDouble(short s) {
    // Convert to range from -1 to (just below) 1
    return s / 32768.0;
}

/**
Read the tones, and call FFT convolve on them
*/
void readTone(char *sampleTone, char *impulseTone){
    FILE *sampleFileStream = fopen(sampleTone, "rb");
    FILE *impulseFileStream = fopen(impulseTone, "rb");
    FILE *outputFileStream = fopen("output.wav", "wb");

    WavHeader header_sample;
    WavHeader header_impulse;
    // read the header subchunk 1, write the header into a new file
    fread(&header_sample, sizeof(header_sample), 1, sampleFileStream);
    fread(&header_impulse, sizeof(header_impulse), 1, impulseFileStream);

    if (header_sample.subchunk1_size != 16){
        // eliminate Null Bytes
        int remainder = header_sample.subchunk1_size -16;
        char randomVar[remainder];
        fread(randomVar, remainder, 1, sampleFileStream);
    }
    
    if (header_impulse.subchunk1_size != 16){
        // eliminate Null Bytes
        int remainder = header_impulse.subchunk1_size -16;
        char randomVar[remainder];
        fread(randomVar, remainder, 1, impulseFileStream);
    }
    char subchunk2_id_sample[4];
    char subchunk2_id_impulse[4];
    int subchunk2_size_sample; // an integer is 4 bytes
    int subchunk2_size_impulse; // an integer is 4 bytes
    fread(&subchunk2_id_sample, sizeof(subchunk2_id_sample), 1, sampleFileStream);
    fread(&subchunk2_size_sample, sizeof(subchunk2_size_sample), 1, sampleFileStream);
    fread(&subchunk2_id_impulse, sizeof(subchunk2_id_impulse), 1, impulseFileStream);
    fread(&subchunk2_size_impulse, sizeof(subchunk2_size_impulse), 1, impulseFileStream);

    int num_samples = subchunk2_size_sample / (header_sample.bits_per_sample / 8); // number of data points in the sample
    int num_impulse = subchunk2_size_impulse / (header_impulse.bits_per_sample / 8); // number of data points in the impulse

    // Allocate memory for the arrays
    int K = next_power_of_2(num_samples * 2);
    double *sample_data = (double *) calloc(K*2, sizeof(double));
    double *impulse_data = (double *) calloc(K*2, sizeof(double));

    int i, ii, j;
    short *s_arr = (short *) malloc((num_samples+50) * sizeof(short));
    short *i_arr = (short *) malloc((num_impulse+50) * sizeof(short));
    fread(&s_arr[0], sizeof(short), num_samples, sampleFileStream);
    fread(&i_arr[0], sizeof(short), num_impulse, impulseFileStream);

    // read the data into the calloc arrays
    for (i = 0, ii = 0; i < num_samples; i++, ii += 2){
        sample_data[ii] = shortToDouble(s_arr[i]);
    }

    for (i = 0, ii = 0; i < num_impulse; i++, ii += 2){
        impulse_data[ii] = shortToDouble(i_arr[i]);
    }

    four1(sample_data - 1, K, 1);
    four1(impulse_data - 1, K, 1);
    
    // float output_data[num_samples + num_impulse - 1];
    double *output_data = (double *) calloc((num_samples + num_impulse - 1) * 2, sizeof(double));

    convolution(sample_data, K, impulse_data, output_data);

    four1(output_data - 1, K, -1);
    
    WavHeader output_header = {
        .chunk_id = {'R', 'I', 'F', 'F'},
        .chunk_size = 36 + (num_samples + num_impulse - 1) * 2,
        .format = {'W', 'A', 'V', 'E'},
        .subchunk1_id = {'f', 'm', 't', ' '},
        .subchunk1_size = 16,
        .audio_format = header_sample.audio_format,
        .num_channels = header_sample.num_channels,
        .sample_rate = header_sample.sample_rate,
        .byte_rate = header_sample.byte_rate,
        .block_align = header_sample.block_align,
        .bits_per_sample = header_sample.bits_per_sample
    };

    int subchunk2_size_output = (num_samples + num_impulse - 1) * 2;

    fwrite(&output_header, sizeof(output_header), 1, outputFileStream);
    fwrite(&subchunk2_id_sample, sizeof(subchunk2_id_sample), 1, outputFileStream);
    fwrite(&subchunk2_size_output, sizeof(subchunk2_size_output), 1, outputFileStream);
    
    double max = -1000000.0;

    for (int i = 0; i < (num_samples + num_impulse - 1) * 2; i += 2) {
        max = abs(output_data[i]) > max ? abs(output_data[i]) : max;
    }

    for (int i = 0; i < (num_samples + num_impulse - 1) * 2; i += 2) {
        short s = (short) ((output_data[i] / max) * 32767.0);
        // printf("Number %d: %d\n", i, s);
        fwrite(&s, sizeof(s), 1, outputFileStream);
    }
}

// main line of execution
int main (int argc, char *argv[]){
    char *sampleTone = NULL;
    char *impulseTone = NULL;

    /*  Process the command line arguments  */
    if (argc == 3) {
        /*  Set a pointer to the output filename  */
        sampleTone = argv[1]; impulseTone = argv[2];
    }
    else {
        /*  The user did not supply the correct number of command-line
            arguments.  Print out a usage message and abort the program.  */
        fprintf(stderr, "Usage:  %s sampleTone impulseTone\n", argv[0]); exit(-1);
    }

    readTone(sampleTone, impulseTone);
}