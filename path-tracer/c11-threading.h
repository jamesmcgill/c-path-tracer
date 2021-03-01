#ifdef __STDC_NO_THREADS__
#error no c11 threading support
#endif

#ifdef __STDC_NO_ATOMICS__
#error no c11 atomics support
#endif

#include <stdatomic.h>
#include <threads.h>

//------------------------------------------------------------------------------
i32 locked_increment_and_return_previous(volatile i32* addend)
{
    return atomic_fetch_add(addend, 1);
}

//------------------------------------------------------------------------------
void sleep_for_ms(i32 sleep_time_ms)
{
    struct timespec ts = { .tv_nsec = sleep_time_ms * 1000000 };
    int res = thrd_sleep(&ts, NULL);
}

//------------------------------------------------------------------------------
typedef struct JobQueue JobQueue;
void process_jobs_until_queue_empty(JobQueue*);

int thread_entry(void* parameter)
{
    JobQueue* job_queue = (JobQueue*)parameter;
    process_jobs_until_queue_empty(job_queue);
    return 0;
}

//------------------------------------------------------------------------------
void dispatch_jobs(JobQueue* job_queue, const i32 max_threads)
{
    thrd_t threads[max_threads];
    for (i32 i = 0; i < max_threads; ++i)
    {
        thrd_create(&threads[i], thread_entry, job_queue);
    }
    for (i32 i = 0; i < max_threads; ++i)
    {
        thrd_join(threads[i], 0);
    }
}

//------------------------------------------------------------------------------
