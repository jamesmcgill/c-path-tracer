#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "windows.h"

//------------------------------------------------------------------------------
i32 locked_increment_and_return_previous(volatile i32* addend)
{
    return InterlockedExchangeAdd(addend, 1);
}

//------------------------------------------------------------------------------
void sleep_for_ms(i32 sleep_time_ms)
{
    Sleep(sleep_time_ms);
}

//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
typedef struct JobQueue JobQueue;
void process_jobs_until_queue_empty(JobQueue*);

DWORD WINAPI win_thread_entry(void* lpParameter)
{
    JobQueue* job_queue = (JobQueue*)lpParameter;
    process_jobs_until_queue_empty(job_queue);
    return 0;
}

//------------------------------------------------------------------------------
void dispatch_jobs(JobQueue* job_queue, const i32 max_threads)
{
    for (i32 i = 0; i < max_threads; ++i)
    {
        DWORD thread_id;
        HANDLE h_thread =
            CreateThread(0, 0, win_thread_entry, job_queue, 0, &thread_id);
        if (h_thread)
        {
            CloseHandle(h_thread);
        }
    }
}

//------------------------------------------------------------------------------
