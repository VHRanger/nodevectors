//  Compile Flage:
//   -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
//
//  Test Script:
//    clear && cc ggvec.c -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic && ./a.out
//
//  TODO: 
//       - get working on karateclub
// 			- remove vocab file dependency
//

// silence the many complaints from visual studio
#define _CRT_SECURE_NO_WARNINGS

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// windows pthread.h is buggy, but this #define fixes it
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash
typedef double real;

//
// TODO: refactor these to be parameters in ggvec_main
//       keep underscored versions here for argv reading
//
int num_threads = 8;
int vector_size = 50; // Embedding components (vector size)
int max_iter = 50; // max number of full passes through data
int verbose = 2; // 0, 1, or 2
int seed = 0; // rng seed
char input_file[MAX_STRING_LENGTH]; // String for input file name
long long n_nodes; // alias of vocab_size in GLoVe
long long *lines_per_thread;
// loss exponent parameter
real exponent = 0.33;
// maximum loss. Can be changed for graphs with large weight values
real x_max = 25.0;
real *W;

// Efficient string comparison
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return (*s1 - *s2);
}

// Find input arguments
int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

// logs errors when loading files.  call after a failed load
#ifdef _MSC_VER
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_s((BUF), (BUFSIZE), (ERRNO))
#else
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_r((ERRNO), (BUF), (BUFSIZE))
#endif
int log_file_loading_error(char *file_description, char *file_name) {
    fprintf(stderr, "Unable to open %s %s.\n", file_description, file_name);
    fprintf(stderr, "Errno: %d\n", errno);
    char error[MAX_STRING_LENGTH];
    STRERROR(errno, error, MAX_STRING_LENGTH);
    fprintf(stderr, "Error description: %s\n", error);
    return errno;
}


//********************************//
//                                //
//       CORE TRAINING LOOP       //
//                                //
//********************************//
void initialize_parameters(uint64_t n_nodes) {
    // TODO: return an error code when an error occurs, clean up in the calling routine
    if (seed == 0) {
        seed = time(0);
    }
    fprintf(stderr, "Using random seed %d\n", seed);
    srand(seed);
    long long a;
    // Buffer size for all the weights
    // 2x number of nodes for context and root node
    // + 1 to allocate space for bias
    long long W_size = 2 * n_nodes * (vector_size + 1);
    a = posix_memalign((void **)&W, 128, W_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for Weights\n");
        exit(1);
    }
    // Initialize new parameters
    for (a = 0; a < W_size; ++a) {
        W[a] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }
}


/* Save params to file */
int save_params() {
    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH+20], output_file_gsq[MAX_STRING_LENGTH+20];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    if (NULL == word) {
        return 1;
    }
    FILE *fid, *fout;
    FILE *fgs = NULL;

    sprintf(output_file,"%s.txt",save_W_file);
    fout = fopen(output_file,"wb");
    if (fout == NULL) {log_file_loading_error("weights file", save_W_file); free(word); return 1;}
    fid = fopen(vocab_file, "r");
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    if (fid == NULL) {log_file_loading_error("vocab file", vocab_file); free(word); fclose(fout); return 1;}
    if (write_header) fprintf(fout, "%lld %d\n", vocab_size, vector_size);
    for (a = 0; a < vocab_size; a++) {
        if (fscanf(fid,format,word) == 0) {free(word); fclose(fid); fclose(fout); return 1;}
        // input vocab cannot contain special <unk> keyword
        if (strcmp(word, "<unk>") == 0) {free(word); fclose(fid); fclose(fout);  return 1;}
        fprintf(fout, "%s",word);
        if (model == 0) { // Save all parameters (including bias)
            for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
        }
        if (model == 1) // Save only "word" vectors (without bias)
            for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
        if (model == 2) // Save "word + context word" vectors (without bias)
            for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
        fprintf(fout,"\n");
        if (fscanf(fid,format,word) == 0) {
            // Eat irrelevant frequency entry
            fclose(fout);
            fclose(fid);
            free(word); 
            return 1;
        }
    }
    fclose(fid);
    fclose(fout);
    if (save_gradsq > 0) fclose(fgs);	
    free(word);
    return 0;
}


/* Train the GGVec model */
void *ggvec_thread(void *vid) {
    long long a, b ,l1, l2;
    long long id = *(long long*)vid;
    CREC cr;
    real diff, fdiff, temp1, temp2;
    FILE *fin;
    fin = fopen(input_file, "rb");
    if (fin == NULL) {
        // TODO: exit all the threads or somehow mark that glove failed
        log_file_loading_error("input file", input_file);
        pthread_exit(NULL);
    }
    //Threads spaced roughly equally throughout file
    // TODO: Make utility to get start/end locations in file
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET);

    real* W_updates1 = (real*)malloc(vector_size * sizeof(real));
    if (NULL == W_updates1){
        fclose(fin);
        pthread_exit(NULL);
    }
    real* W_updates2 = (real*)malloc(vector_size * sizeof(real));
        if (NULL == W_updates1){
        fclose(fin);
        free(W_updates1);
        pthread_exit(NULL);
    }
    for (a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        if (cr.word1 < 1 || cr.word2 < 1) { continue; }
        
        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words
        
        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff

        // Check for NaN and inf() in the diffs.
        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)) {
            fprintf(stderr,"Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }

        cost[id] += 0.5 * fdiff * diff; // weighted squared error
        
        /* Adaptive gradient updates */
        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        for (b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = fmin(fmax(fdiff * W[b + l2], -grad_clip_value), grad_clip_value) * eta;
            temp2 = fmin(fmax(fdiff * W[b + l1], -grad_clip_value), grad_clip_value) * eta;
            // adaptive updates
            W_updates1[b] = temp1 / sqrt(gradsq[b + l1]);
            W_updates2[b] = temp2 / sqrt(gradsq[b + l2]);
            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
        if (!isnan(W_updates1_sum) && !isinf(W_updates1_sum) && !isnan(W_updates2_sum) && !isinf(W_updates2_sum)) {
            for (b = 0; b < vector_size; b++) {
                W[b + l1] -= W_updates1[b];
                W[b + l2] -= W_updates2[b];
            }
        }

        // updates for bias terms
        W[vector_size + l1] -= check_nan(fdiff / sqrt(gradsq[vector_size + l1]));
        W[vector_size + l2] -= check_nan(fdiff / sqrt(gradsq[vector_size + l2]));
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
        
    }
    free(W_updates1);
    free(W_updates2);
    
    fclose(fin);
    pthread_exit(NULL);
}


int train_ggvec()
    {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin;
    real total_cost = 0;
    long long num_lines;
    long long *thread_splits;

    // TODO: Add loop to find n_nodes if not supplied
    //       Also fill node map on first loop
    n_nodes = fill_nodemap();
    // need to split file at newlines
    // GLoVe has fixed-length lines
    // Get inspired by wc impl here: 
    //    https://stackoverflow.com/questions/17925051/fast-textfile-reading-in-c
    //    https://unix.stackexchange.com/questions/266887/need-something-that-is-faster-than-wc-l
    num_lines = fill_nodemap();
    thread_splits = fill_nodemap();
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
    lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
    //
    // \end file splitting for threads
    //
    if (verbose > 0) fprintf(stderr, "TRAINING MODEL\n");
    fin = fopen(input_file, "rb");
    if (fin == NULL) {log_file_loading_error("Edgelit file", input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters(n_nodes);
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"number of nodes: %lld\n", n_nodes);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr,"exponent: %lf\n", exponent);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    // Lock-Free SGD Implementation
    for (b = 0; b < max_iter; b++) {
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        // TODO: creates and joins threads every iteration...
        //       should use threadpool instead
        //       using OPENMP is probably best here
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, ggvec_thread, (void *)&thread_ids[a]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        free(thread_ids);

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, cost: %lf\n", time_buffer,  b+1, total_cost/num_lines);
    }
    free(pt);
    free(lines_per_thread);
    return save_params(-1);
}


int main(int argc, char **argv) {
    int i;
    FILE *fid;
    int result = 0;

    if (argc == 1) {
        printf("GGVec: Global Graph Vectors, v0.1.16\n");
        printf("Author: Matt Ranger (matthieu(dot)ranger(at)outlook(dot)com)\n\n");
        printf("Usage options:\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
    } else {
        // TODO: Make this also the default empty argument
        if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
        // TODO: Raise and error and exit if no input file
        else strcpy(input_file, (char *)"--------.fail.insert.error.here.------------");
        if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        result = train_ggvec();
    }
    free(W);
    return result;
}
