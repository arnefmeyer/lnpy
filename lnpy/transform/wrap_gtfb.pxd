
cdef extern from "src/gammatone/Gfb_analyze.h":
       void analyze(unsigned int bands,
                    unsigned int gamma,
                    unsigned int samples,
                    double * real_filter_coefficients,
                    double * imag_filter_coefficients,
                    double * normalization_factors,
                    double * real_filter_states,
                    double * imag_filter_states,
                    double * real_input,
                    double * real_output,
                    double * imag_output)
