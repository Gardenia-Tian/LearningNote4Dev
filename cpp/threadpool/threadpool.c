#include<stdio.h>
#include<pthread.h> 
#include<assert.h>

// 当成一个函数在用
#define LINK_ADD(item,list) do{ \
    item->next = list;          \
    item->prev = NULL;          \
    if(list!=NULL) list->prev = item; \
    list = item;                \
} while(0)                      \


#define LINK_RMV(item, list) do{    \
    if(item->prev!=NULL) item->prev->next = item->next;  \
    if(item->next!=NULL) item->next->prev = item->prev;  \
    if(item == list) list = item->next;                  \
    item->next = NULL;              \
    item->prev = NULL;              \
} while(0)                          \


struct worker_entry{
    struct worker_entry *next;
    struct worker_entry *prev;
    struct thread_pool* pool;
    
    pthread_t id;
    // 控制线程是否退出
    int terminate;
    
};

// 每一个任务
struct task_entry{
#if 0
    struct task_entry* next;
#else 
    struct task_entry *next;
    struct task_entry *prev;
#endif 
    // 这里是声明了一个函数指针, 这个指针指向一个没有返回值的函数
    void (*handler)(void* arg);
    void *userdata;
};

// 这种做法搭配上面的if 0食用
#if 0
struct task_queue{
    struct task_entry* head;
    // 二级指针, 更方便加入和移除, 二级指针可以一下子控制两个(尾巴和尾巴后面的那个)
    struct task_entry **tail;
};
#endif


struct thread_pool{
    struct worker_entry *worker_queue;
    struct task_entry* task_queue;

    // 条件等待
    pthread_cond_t cond;
    // 互斥锁
    pthread_mutex_t mutex;
};

void thread_task_cycle(void* arg){
    struct worker_entry* worker = (struct worker_entry*)arg;

    while(worker->terminate){
        pthread_mutex_lock(&worker->pool->mutex);
        // 如果现在还没有任务可以分配, 那就等着
        while(worker->pool->task_queue == NULL){
            pthread_cond_wait(&worker->pool->cond, &worker->pool->mutex);
        }
        struct task_entry* task = worker->pool->task_queue; 
        // 给这个worker分配任务, 对队列的操作要加锁
        LINK_RMV(task, worker->pool->task_queue);
        
        pthread_mutex_unlock(&worker->pool->mutex);

        task->handler(task->userdata);
    }
   
}


int thread_pool_setup(struct thread_pool* pool, int num){
    assert(pool != NULL);
    if(num < 1) num = 1;
    for(int idx = 0;idx<num;idx++){
        struct worker_entry* worker = (struct worker_entry*)malloc(sizeof(struct worker_entry));
        if(worker == NULL){
            perror("malloc");
            continue;
        }
        memset(worker,0, sizeof(struct worker_entry));
        worker->pool = pool;
        pthread_create(&worker->id, NULL, thread_task_cycle, worker);
        // 让出主线程
        usleep(1);
        LINK_ADD(worker, pool->worker_queue);
    }
}

// read(); accept(); write(); epoll(-1); recv(); send();

// 条件满足才返回
// pthread_cond_wait()
// 条件或超时都返回
// pthread_cond_waittime()

void task_pool_push_task(struct thread_pool* pool, struct task_entry* task){
    pthread_mutex_lock(&pool->mutex);
    LINK_ADD(task,pool->task_queue);
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
}