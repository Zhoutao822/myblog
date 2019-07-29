---
title: Android框架-RxJava
date: 2019-07-25 20:32:26
category:
- Android
tags:
- Android
- Handler
- AsyncTask
- EventBus
- RxJava
- 多线程
---

参考：

> [RxJava2](https://github.com/ReactiveX/RxJava/tree/2.x)
> [EventBus](http://greenrobot.org/eventbus/)
> [Android Handler 消息机制详述](https://www.jianshu.com/p/1a5a3db45cfa)
> [Android 多线程：手把手教你使用AsyncTask](https://www.jianshu.com/p/ee1342fcf5e7)
> [EventBus使用详解](https://juejin.im/post/5a6c36fff265da3e2f012f82)
> [Rxjava这一篇就够了，墙裂推荐](https://juejin.im/post/5a224cc76fb9a04527256683)

Android中很多地方都需要跨线程通信，这是由于Android主线程不允许进行复杂的网络请求或者其他非常耗时的操作，否则会导致ANR，主线程只能进行UI操作，比如修改某个控件的text、设置某个控件不可见等等，因此网络请求等操作需要在其他线程中完成，当数据在其他线程中获取完毕时，通过跨线程通信将数据传到主线程中，主线程就可以直接根据数据进行UI操作。常见的跨线程通信的方式有Handler、AsyncTask、EventBus以及RxJava等，前两个是Android自带，后两者是封装好的第三方库。

<!-- more -->

## 1. Handler

Handler是Android中最简单的线程间通信方式，同时也可以在同一个线程中发送消息，但是使用时需要注意内存泄漏的问题。

### 1.1 Handler简单使用

还是以和风天气请求为例，我们的目标是在子线程中请求数据，然后通过Handler将数据传到主线程中并显示出来。

```java
public class MainActivity extends AppCompatActivity {
    private final static String KEY = "XXXXXXXXXX";
    private final static String URL = "https://free-api.heweather.net/s6/weather/";

    private TextView textView;
    private Handler handler;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textView);
        // Handler的实例化，重写handleMessage方法用于等待处理msg，
        // handleMessage方法是回调，在回调中更新UI，此时执行在主线程，
        // 在Android Studio中会提示这里存在内存泄漏问题
        handler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                textView.setText(msg.obj.toString());
            }
        };
        // 在子线程开启一个网络请求
        new Thread(new Runnable() {
            @Override
            public void run() {
                // Retrofit通用代码
                Retrofit retrofit = new Retrofit.Builder()
                        .baseUrl(URL) // 设置网络请求的公共Url地址
                        .addConverterFactory(GsonConverterFactory.create()) // 设置数据解析器
                        .build();

                Api api = retrofit.create(Api.class);
                Call<WeatherEntity> call = api.getNowWeather("beijing", KEY);
                try {
                    // 为了在当前子线程获取数据，这里直接使用execute
                    WeatherEntity result = call.execute().body();
                    // Message的实例化方法Message.obtain
                    Message message = Message.obtain();
                    // 可以通过Message附加很多数据，这里仅用obj，保存我们网络请求得到的实例
                    message.obj = result;
                    // 通过handler.sendMessage(message)实现调用回调方法，完成数据传输
                    // 这种操作有点类似于接口回调
                    handler.sendMessage(message);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
}
```

这里的内存泄露的原因可以参考其他资料，主要是**Java 中非静态内部类和匿名内部类会持有外部类的引用**同时**Handler 的生命周期比外部类长**导致的。如何解决，肯定就是让Handler是静态内部类就完事了

```java
public class MainActivity extends AppCompatActivity {
    private final static String KEY = "XXXXXXXXXX";
    private final static String URL = "https://free-api.heweather.net/s6/weather/";

    private TextView textView;
    private Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textView);
        // 使用自定义的静态内部类
        handler = new MyHandler(this);
        // 子线程请求没有变化
        new Thread(new Runnable() { 
            @Override
            public void run() {
                Retrofit retrofit = new Retrofit.Builder()
                        .baseUrl(URL) // 设置网络请求的公共Url地址
                        .addConverterFactory(GsonConverterFactory.create()) // 设置数据解析器
                        .build();

                Api api = retrofit.create(Api.class);
                Call<WeatherEntity> call = api.getNowWeather("beijing", KEY);
                try {
                    WeatherEntity result = call.execute().body();
                    Message message = Message.obtain();
                    message.obj = result;
                    handler.sendMessage(message);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacksAndMessages(null);
        super.onDestroy();
    }
    // 
    public void handleMessage(Message msg) {
        textView.setText(msg.obj.toString());
    }
    // 自定义静态内部类，与onDestroy中removeCallbacksAndMessages一起使用
    public static class MyHandler extends Handler {
        private WeakReference<MainActivity> reference;

        public MyHandler(MainActivity mainActivity) {
            // 同时需要持有对MainActivity的弱引用
            this.reference = new WeakReference<>(mainActivity);
        }

        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            MainActivity mainActivity = reference.get();
            if (mainActivity != null) {
                // 将msg传给MainActivity处理
                mainActivity.handleMessage(msg);
            }
        }
    }
}
```

### 1.2 Handler线程间通信

上面的例子仅演示了从子线程传数据给主线程，那么如果同时需要从主线程传数据给子线程，怎么办

```java
// 首先需要自定义MyThread，完成Looper的初始化，否则子线程不会自动初始化Looper
    private class MyThread extends Thread {
        private Looper looper;

        @Override
        public void run() {
            super.run();
            Looper.prepare();
            looper = Looper.myLooper();
            Looper.loop();
        }
    }

// 然后在onCreate方法中
    Log.i("aaaa", String.valueOf(Thread.currentThread()));
    MyThread thread = new MyThread();
    thread.start(); // 必须先启动子线程
    while (true) {
        // 确保子线程中的Looper初始化完成
        if (thread.looper != null) {
            // 此时handler的handleMessage方法是在子线程MyThread中执行的
            // 两处log中线程的值是不一样的，通过Handler的构造方法实现子线程的调用
            handler = new Handler(thread.looper) {
                @Override
                public void handleMessage(Message msg) {
                    Log.i("aaaa", String.valueOf(msg.what) + Thread.currentThread());
                }
            };
            handler.sendEmptyMessage(12321);
            break;
        }
    }

// 如果使用定义好的HandlerThread，则不需要继承Thread，直接使用，
// HandlerThread默认帮我们完成了Looper的初始化
    Log.i("aaaa", String.valueOf(Thread.currentThread()));
    // HandlerThread需要用String的构造方法，我们在log中也可以看到
    HandlerThread thread = new HandlerThread("new thread");
    thread.start();
    while (true) {
        if (thread.getLooper() != null) {
            handler = new Handler(thread.getLooper()) {
                @Override
                public void handleMessage(Message msg) {
                    Log.i("aaaa", String.valueOf(msg.what) + Thread.currentThread());
                }
            };
            handler.sendEmptyMessage(12321);
            break;
        }
    }
```

> 为什么子线程需要初始化Looper，而主线程不需要？

首先需要明白的是，只有需要处理消息的线程才需要Looper，即哪个线程执行了handleMessage方法，则线程需要Looper，原因在源码分析中解释；主线程以及HandlerThread会自动进行Looper的初始化，而`new Thread()`不会，因此在第二个例子中，子线程需要处理消息，所以需要初始化Looper而第一个例子中主线程不需要。

> Handler的初始化，其构造方法依赖于什么，为什么第二个例子中Handler不是在主线程中初始化的吗？

首先需要知道的是Handler是可以被跨线程调用的，而View是不可以的，举个例子，如果在第一个例子中我们在子线程中调用`textView.setText(result.toString());`，则会报错`CalledFromWrongThreadException: Only the original thread that created a view hierarchy can touch its views.`，而Handler没问题，Handler默认构造方法`new Handler()`会将当前线程的Looper保存在自己这个实例中，即将主线程中的Looper保存，而带参数的构造方法`new Handler(thread.looper)`会保存thread的looper在实例中，又因为Handler是可以跨线程调用的，所以区分Handler属于哪个线程其实是根据构造方法传入的参数决定的，至于Handler归属于不同的线程会有什么影响，在源码分析中解释。

### 1.3 Handler源码分析

以从子线程向主线程发送消息为例，首先从ActivityThread的main方法开始，前面说过主线程中的Looper是自动初始化的，其初始化的位置就在ActivityThread的main方法中

```java
// ActivityThread.java 核心就两个Looper.prepareMainLooper()和Looper.loop()
    public static void main(String[] args) {
      
        // ...

        Looper.prepareMainLooper();

        // ...
        // 显然这里是不会执行的
        if (false) {
            Looper.myLooper().setMessageLogging(new
                    LogPrinter(Log.DEBUG, "ActivityThread"));
        }

        // End of event ActivityThreadMain.
        Trace.traceEnd(Trace.TRACE_TAG_ACTIVITY_MANAGER);
        Looper.loop();

        throw new RuntimeException("Main thread loop unexpectedly exited");
    }
```

再看看Looper.prepareMainLooper()的作用

```java
// Looper.java 看注释就知道是是为主线程初始化Looper，关键还是看prepare方法，再看myLooper
    /**
     * Initialize the current thread as a looper, marking it as an
     * application's main looper. The main looper for your application
     * is created by the Android environment, so you should never need
     * to call this function yourself.  See also: {@link #prepare()}
     */
    public static void prepareMainLooper() {
        prepare(false);
        synchronized (Looper.class) {
            if (sMainLooper != null) {
                throw new IllegalStateException("The main Looper has already been prepared.");
            }
            sMainLooper = myLooper();
        }
    }

// prepare方法通过sThreadLocal set了一个Looper实例，
// 一个Looper实例保存了MessageQueue和Thread.currentThread()
    private static void prepare(boolean quitAllowed) {
        if (sThreadLocal.get() != null) {
            throw new RuntimeException("Only one Looper may be created per thread");
        }
        sThreadLocal.set(new Looper(quitAllowed));
    }

// myLooper方法从sThreadLocal get到Looper，那正好对应上面prepare set的Looper，
// ThreadLocal的作用是可以保存线程内的变量，简而言之就是通过ThreadLocal的set和get方法
// 处理的变量仅属于某个线程，以Looper为例，在某个线程中有且仅有一个
    /**
     * Return the Looper object associated with the current thread.  Returns
     * null if the calling thread is not associated with a Looper.
     */
    public static @Nullable Looper myLooper() {
        return sThreadLocal.get();
    }   

// 最后调用了Looper.loop()
    /**
     * Run the message queue in this thread. Be sure to call
     * {@link #quit()} to end the loop.
     */
    public static void loop() {
        // Looper.loop()会进入一个死循环，但是这个循环并不会导致卡死，
        // 涉及到Linux pipe/epoll机制，简单说就是在主线程的MessageQueue没有消息时，
        // 便阻塞在loop的queue.next()中的nativePollOnce()方法里，此时主线程会释放CPU资源进入休眠状态，
        // 直到下个消息到达或者有事务发生，通过往pipe管道写端写入数据来唤醒主线程工作。
        // 这里采用的epoll机制，是一种IO多路复用机制，可以同时监控多个描述符，
        // 当某个描述符就绪(读或写就绪)，则立刻通知相应程序进行读或写操作，本质同步I/O，即读写是阻塞的。
        //  所以说，主线程大多数时候都是处于休眠状态，并不会消耗大量CPU资源。
        // 先拿到当前线程的Looper，然后拿到Looper中的MessageQueue
        final Looper me = myLooper();
        if (me == null) {
            throw new RuntimeException("No Looper; Looper.prepare() wasn't called on this thread.");
        }
        final MessageQueue queue = me.mQueue;

        // ...
        // 开启循环，Android中主线程上所有的点击事件、UI绘制都是通过Message发送到MessageQueue中等待执行
        // 所以这里必须是死循环，因为如果跳出了这个循环说明已经无法再继续处理任何Message，那么随之而来的肯定就是
        // 应用崩溃或者重启Looper，但是这里的循环并不会导致卡死，理由在上面已经简要说明了
        for (;;) {
            // 循环的作用就是通过queue.next()不断地从MessageQueue取出Message，next方法中也是一个死循环，
            // 正常情况下queue.next()应该返回一个有效的Message，或者休眠不返回任何值，如果返回null，
            // 说明出了问题
            Message msg = queue.next(); // might block
            if (msg == null) {
                // 当取出的message为空时说明MessageQueue被终止了，因此跳出循环，执行其他操作，比如重启Looper或者崩溃？
                // No message indicates that the message queue is quitting.
                return;
            }

            // ...
            // 当我们取到有效的Message后，就需要知道这个Message应该由谁来处理，即Target，从Message源码中可知，
            // 这个Target实际上就是Handler，最终调用的就是Handler的dispatchMessage方法，从这里我们就知道了
            // 只要其他线程能够将Message发送到主线程的MessageQueue中，那么这个Message就可以被主线程的Handler处理
            try {
                msg.target.dispatchMessage(msg);
                dispatchEnd = needEndTime ? SystemClock.uptimeMillis() : 0;
            } finally {
                if (traceTag != 0) {
                    Trace.traceEnd(traceTag);
                }
            }

            // ...
            // 最后需要对Message对象进行回收
            msg.recycleUnchecked();
        }
    }
```

ActivityThread的main方法中对主线程的Looper进行初始化，同样的主线程的MessageQueue也准备好对其中的Message进行分发，这都是通过死循环实现的，相当于MessageQueue是一个等待队列，有消息来了，他就取消息并调用Message对应的Handler的dispatchMessage方法，如果没有就休眠，然后我们看看Handler的初始化以及Message的发送是如何实现的

```java
// Handler.java Handler的构造方法分为两类，一类是参数带Looper的，另一类是不带Looper
// 不带Looper的构造函数最终会调用到最后一个构造函数，并进行Looper的初始化；
// 带Looper的构造函数会直接保存参数中的Looper实例
    public Handler() {
        this(null, false);
    }

    public Handler(Callback callback) {
        this(callback, false);
    }

    public Handler(Looper looper) {
        this(looper, null, false);
    }

    public Handler(Looper looper, Callback callback) {
        this(looper, callback, false);
    }

    public Handler(boolean async) {
        this(null, async);
    }

    public Handler(Looper looper, Callback callback, boolean async) {
        mLooper = looper;
        mQueue = looper.mQueue;
        mCallback = callback;
        mAsynchronous = async;
    }

    public Handler(Callback callback, boolean async) {
        if (FIND_POTENTIAL_LEAKS) {
            final Class<? extends Handler> klass = getClass();
            if ((klass.isAnonymousClass() || klass.isMemberClass() || klass.isLocalClass()) &&
                    (klass.getModifiers() & Modifier.STATIC) == 0) {
                Log.w(TAG, "The following Handler class should be static or leaks might occur: " +
                    klass.getCanonicalName());
            }
        }
        // Looper的myLooper方法会初始化当前线程的Looper
        mLooper = Looper.myLooper();
        if (mLooper == null) {
            throw new RuntimeException(
                "Can't create handler inside thread " + Thread.currentThread()
                        + " that has not called Looper.prepare()");
        }
        mQueue = mLooper.mQueue;
        mCallback = callback;
        mAsynchronous = async;
    }
```

然后调用`handler.sendMessage(message);`

```java
// Handler.java sendMessage方法会直接调用sendMessageDelayed
// sendMessageDelayed就是多个延时的效果
    public final boolean sendMessage(Message msg)
    {
        return sendMessageDelayed(msg, 0);
    }

    public final boolean sendMessageDelayed(Message msg, long delayMillis)
    {
        if (delayMillis < 0) {
            delayMillis = 0;
        }
        // 通过加上SystemClock.uptimeMillis()可以直接得到执行的具体时间
        return sendMessageAtTime(msg, SystemClock.uptimeMillis() + delayMillis);
    }

    public boolean sendMessageAtTime(Message msg, long uptimeMillis) {
        MessageQueue queue = mQueue;
        if (queue == null) {
            RuntimeException e = new RuntimeException(
                    this + " sendMessageAtTime() called with no mQueue");
            Log.w("Looper", e.getMessage(), e);
            return false;
        }
        // 最终还是使用Handler的MessageQueue
        return enqueueMessage(queue, msg, uptimeMillis);
    }

    private boolean enqueueMessage(MessageQueue queue, Message msg, long uptimeMillis) {
        // 注意这里将Message的target设置为当前handler
        msg.target = this;
        if (mAsynchronous) {
            msg.setAsynchronous(true);
        }
        // 然后调用MessageQueue的enqueueMessage方法
        return queue.enqueueMessage(msg, uptimeMillis);
    }

// MessageQueue.java enqueueMessage将Message加入链表中
    boolean enqueueMessage(Message msg, long when) {
        if (msg.target == null) {
            throw new IllegalArgumentException("Message must have a target.");
        }
        if (msg.isInUse()) {
            throw new IllegalStateException(msg + " This message is already in use.");
        }

        synchronized (this) {
            // 如果MessageQueue被终止了，那么Message还需要回收
            if (mQuitting) {
                IllegalStateException e = new IllegalStateException(
                        msg.target + " sending message to a Handler on a dead thread");
                Log.w(TAG, e.getMessage(), e);
                msg.recycle();
                return false;
            }

            msg.markInUse();
            msg.when = when;
            Message p = mMessages;
            boolean needWake;
            // 根据msg.next基本可以发现Message是一个链表中的节点，也就是说MessageQueue中的mMessages
            // 是一种链表形式的结构，其中mMessages是表头，当执行next方法时就会将表头也就是mMessages表示的
            // Message返回，当我们传入的Message满足以下任意条件时，可以将此Message作为表头：
            // 1. 表头本身为空，很明显当没有任何Message传入的时候；
            // 2. 当我们传入的Message没有任何延迟，这也很显然，立即执行的Message当然要放第一个；
            // 3. 当我们传入的Message的执行时间在表头的执行时间之前，这也很显然，按照时间排序。
            if (p == null || when == 0 || when < p.when) {
                // New head, wake up the event queue if blocked.
                msg.next = p;
                mMessages = msg;
                needWake = mBlocked;
            } else {
                // 如果Message不是表头位置，那么肯定就是链表中的某个位置
                // Inserted within the middle of the queue.  Usually we don't have to wake
                // up the event queue unless there is a barrier at the head of the queue
                // and the message is the earliest asynchronous message in the queue.
                needWake = mBlocked && p.target == null && msg.isAsynchronous();
                Message prev;
                for (;;) {
                    // 链表的遍历，还要判断时间when
                    prev = p;
                    p = p.next;
                    if (p == null || when < p.when) {
                        break;
                    }
                    if (needWake && p.isAsynchronous()) {
                        needWake = false;
                    }
                }
                // 这就很简单了，有序链表中加入某个节点，排序方式为when的值
                msg.next = p; // invariant: p == prev.next
                prev.next = msg;
            }

            // We can assume mPtr != 0 because mQuitting is false.
            if (needWake) {
                nativeWake(mPtr);
            }
        }
        return true;
    }
```

到这里我们就知道了Message被Handler加到了Handler线程的MessageQueue中，而Handler线程中的Looper一直在等待Message进入MessageQueue，通过queue.next()取出Message，然后调用Handler的dispatchMessage方法

```java
    /**
     * Handle system messages here.
     */
    public void dispatchMessage(Message msg) {
        // dispatchMessage处理Message的方式也很简单
        // 首先判断Message是否设置了Callback，如果有
        // 则执行message.callback.run()
        if (msg.callback != null) {
            handleCallback(msg);
        } else {
            // 如果没有，则判断Handler是否初始化设置了Callback，
            // 这个和Handler的构造函数相关
            if (mCallback != null) {
                if (mCallback.handleMessage(msg)) {
                    return;
                }
            }
            // 否则就执行handler重写的handleMessage方法，
            // 这个方法是在我们继承Handler时重写的，或者
            // 在使用Handler匿名内部类时重写的
            handleMessage(msg);
        }
    }
```

以上就是完整的通过Handler从子线程发送消息到主线程并执行的过程，也解决了我的一些问题：

> 1.为什么要设计Handler来传输消息？

因为多线程的情况下并不确定子线程何时能够执行完毕获取数据，所以需要设计Handler实现一种回调机制，即当子线程数据获取完成后将数据传到主线程中，通过主线程中的回调决定如何处理传来的数据。

> 2.为什么要用MessageQueue和Looper这种工具？

我想是因为既然子线程并不确定何时结束，其次如果存在多个子线程向主线程传递消息，那干脆将这些消息都放在一个队列MessageQueue中，因为多个子线程之间的执行顺序我们也无法确定，如果放在队列中，那么根据消息附加的时间来进行排序我们就可以按照顺序读取从各个子线程发送过来的消息了，与此同时，需要一个能够不停地读取队列中消息的工具Looper，Looper可以循环取数据但是不会阻塞卡死。

### 1.4 Handler进阶

Handler除了可以发送Message外，还可以post Runnable，Runnable是接口，提供run方法，Thread类实现了Runnable接口，所以Thread需要实现run方法，run方法中的内容就是执行在Thread线程中，如果Runnable是通过Handler post，那么根据Message的原理，应该明白此Runnable就是运行在Handler归属的线程中

```java
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    textView = findViewById(R.id.textView);
    // handler不重写handleMessage方法
    handler = new Handler();

    new Thread(new Runnable() {
        @Override
        public void run() {
            // Retrofit通用代码
            Retrofit retrofit = new Retrofit.Builder()
                    .baseUrl(URL) // 设置网络请求的公共Url地址
                    .addConverterFactory(GsonConverterFactory.create()) // 设置数据解析器
                    .build();

            Api api = retrofit.create(Api.class);
            Call<WeatherEntity> call = api.getNowWeather("beijing", KEY);
            try {
                // 为了在当前子线程获取数据，这里直接使用execute
                final WeatherEntity result = call.execute().body();
                // 通过post直接修改textView的text
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        textView.setText(result.toString());
                    }
                });
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }).start();
}
```

因为textView只能在主线程进行设置，所以很显然handler post的Runnable是在主线程运行的，这样就不需要传递数据，而是直接处理数据了，下面看看Runnable是如何被处理的

```java
// Handler.java
    public final boolean post(Runnable r)
    {
        // sendMessageDelayed之前分析过，这里是将Runnable放入Message中了
       return  sendMessageDelayed(getPostMessage(r), 0);
    }

// 通过Message保存了这个Runnable，保存在callback，这个之前在dispatchMessage中见过
    private static Message getPostMessage(Runnable r) {
        Message m = Message.obtain();
        m.callback = r;
        return m;
    }   

    public void dispatchMessage(Message msg) {
        // 之前sendMessage都是走的第二个判断，post走的就是第一个判断，
        // 我们的Runnable现在不为空
        if (msg.callback != null) {
            handleCallback(msg);
        } else {
            if (mCallback != null) {
                if (mCallback.handleMessage(msg)) {
                    return;
                }
            }
            handleMessage(msg);
        }
    }
// 结果很明显了，就是执行了Runnable的run方法
    private static void handleCallback(Message message) {
        message.callback.run();
    }
```

下面演示一下多个线程向主线程发送消息会产生怎样的结果，自定义线程MessageThread用于发送Message，普通的Thread用于post Runnable

```java
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textView);
        // handler根据传过来的Message的what值进行不同的操作
        handler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case 0:
                        Log.i("aaaa", "Thread 0: " + msg.obj.toString());
                        break;
                    case 1:
                        Log.i("aaaa", "Thread 1: " + msg.obj.toString());
                        break;
                    case 2:
                        Log.i("aaaa", "Thread 2: " + msg.obj.toString());
                        break;
                }
            }
        };
        // 开启三个线程发送Message，加上延时
        new MessageThread(0, "thread 0 hahaha", 3000).start();
        new MessageThread(1, "thread 1 oooooo", 1000).start();
        new MessageThread(2, "thread 2 yyyyyy", 2000).start();

        // post Runnable也加上延时，注意这里的postDelayed并不会阻塞主线程，
        // 原理同Looper.loop()，所以不会引起ANR，这个延时只会影响此Message在MessageQueue
        // 中的位置
        new Thread(new Runnable() {
            @Override
            public void run() {
                handler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        Log.i("aaaa", "MainThread");
                    }
                }, 10000);
            }
        }).start();

    }

    private class MessageThread extends Thread {

        private int what;
        private String text;
        private long delay;

        public MessageThread(int what, String text, long delay) {
            this.what = what;
            this.text = text;
            this.delay = delay;
        }

        @Override
        public void run() {
            Message message = Message.obtain();
            message.what = what;
            message.obj = text;
            handler.sendMessageDelayed(message, delay);
        }
    }
```

log结果为

```
Thread 1: thread 1 oooooo
Thread 2: thread 2 yyyyyy
Thread 0: thread 0 hahaha
MainThread
```

## 2. AsyncTask

通过Handler实现的多线程通信在使用上还是有很多不方便的地方，比如需要显示的创建子线程，每次创建子线程都是对资源的消耗，当然也可以使用线程池来减少线程资源的创建与销毁，同时需要定义Handler的处理方式，对于每一个需要处理消息的线程都需要定义其Handler，这样就显得比较乱，因此可以使用AsyncTask来替代，先看一下如何使用。

### 2.1 AsyncTask简单使用

```java
// 依然以请求和风天气数据为例，现在我们为加载数据时显示进度，为什么要显示进度呢
// 从设计理念来看，当我们给某些需要长时间等待的操作加上进度条时，用户对这个操作的
// 容忍度会增加，比如常见的进入游戏的界面，会显示进度条，这样的话就算耗时相对较长，
// 但是用户可以根据进度有一个心理预期，从而提升容忍度；如果你的耗时操作没有任何进度
// 提示，那么用户很容易觉得你的应用是不是卡死了，从而降低了体验
public class MainActivity extends AppCompatActivity {

    private static final String KEY = "XXXXXXXXXXXX";
    private static final String URL = "https://free-api.heweather.net/s6/weather/";

    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textView);
        MyTask task = new MyTask();
        // execute的参数等价于doInBackground的参数
        task.execute("beijing", 50L);
    }
// AsyncTask是抽象类，需要自定义Task并实现doInBackground方法，除了doInBackground之外
// 还有几个方法用于初始化、显示进度、输出结果等功能，三个参数Object, Integer, WeatherEntity为泛型参数
    private class MyTask extends AsyncTask<Object, Integer, WeatherEntity> {

        // onPreExecute在主线程执行，用于做一些提前的初始化
        @Override
        protected void onPreExecute() {
            textView.setText("Start!!");
        }

        // doInBackground在子线程执行，不需要显示地创建Thread，这里的参数params
        // 是一个泛型参数，也就是说可以传入多个参数，相当于参数数组，参数的传入是
        // task.execute传入，返回值由AsyncTask第三个泛型参数决定，同时也是
        // onPostExecute的输入参数
        @Override
        protected WeatherEntity doInBackground(Object... params) {
            // 根据传入的顺序读取，location就是beijing，delay就是50L，
            // 为了模拟进度，这里传入一个延时，正式使用时需要根据数据实际传输的进度
            // 展示进度
            String location = (String) params[0];
            long delay = (long) params[1];
            WeatherEntity result = null;
            try {
                // 先获取数据，我们知道这里取数据的速度其实是很快的
                result = getData(location);
            } catch (IOException e) {
                e.printStackTrace();
            }
            // 然后显示进度，这里仅模拟
            try {
                int count = 0;
                int length = 1;
                while (count < 99) {

                    count += length;
                    // 可调用publishProgress（）显示进度, 之后将执行onProgressUpdate（）
                    publishProgress(count);
                    // 模拟耗时任务
                    Thread.sleep(delay);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            // 最后返回取到的数据
            return result;
        }

        // onProgressUpdate在主线程执行，显示进度
        @Override
        protected void onProgressUpdate(Integer... values) {
            textView.setText(String.format(Locale.CHINA, "加载进度：%d%%", values[0]));
        }

        // onPostExecute在主线程执行，用于处理doInBackground返回的数据
        @Override
        protected void onPostExecute(WeatherEntity weatherEntity) {
            textView.setText(weatherEntity.toString());
        }

        // AsyncTask可以通过调用isCancelled主动终止
        @Override
        protected void onCancelled() {
            textView.setText("Cancel!!!");
        }

        private WeatherEntity getData(String location) throws IOException {
            Retrofit retrofit = new Retrofit.Builder()
                    .baseUrl(URL)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
            Api api = retrofit.create(Api.class);
            Call<WeatherEntity> call = api.getNowWeather(location, KEY);
            return call.execute().body();
        }
    }
}
```

### 2.2 AsyncTask源码分析

首先从task.execute开始

```java
// AsyncTask.java
    @MainThread
    public final AsyncTask<Params, Progress, Result> execute(Params... params) {
        // executeOnExecutor传入两个参数sDefaultExecutor和params，
        // sDefaultExecutor看名字就知道是一个Executor，Executor提供execute方法，
        // 用于消耗Runnable，我们先看看sDefaultExecutor是什么
        return executeOnExecutor(sDefaultExecutor, params);
    }

    // sDefaultExecutor实际上是new SerialExecutor()，static修饰加上
    // synchronized修饰execute方法，保证多个Task启动execute时是按照顺序执行的
    private static class SerialExecutor implements Executor {
        // SerialExecutor提供一个队列mTasks用于保存Runnable
        // mActive表示当前需要执行的Runnable
        final ArrayDeque<Runnable> mTasks = new ArrayDeque<Runnable>();
        Runnable mActive;
        // execute方法把传入的Runnable加入到队列中，但是不是直接加入的，
        // 而是通过new Runnable改造了，让其在执行了run之后会执行scheduleNext
        public synchronized void execute(final Runnable r) {
            mTasks.offer(new Runnable() {
                public void run() {
                    try {
                        r.run();
                    } finally {
                        scheduleNext();
                    }
                }
            });
            if (mActive == null) {
                scheduleNext();
            }
        }
        // scheduleNext从mTasks的对头取Runnable，通过THREAD_POOL_EXECUTOR
        // 执行Runnable，联系SerialExecutor的execute方法，就知道了一旦调用了
        // SerialExecutor的execute方法，就会不断从mTasks取任务，然后交给线程池
        // THREAD_POOL_EXECUTOR去执行，至于线程池是如何execute暂时不解释，
        // 只需要知道线程池会分配空闲的线程并执行传入的mFuture的run方法即可
        protected synchronized void scheduleNext() {
            if ((mActive = mTasks.poll()) != null) {
                THREAD_POOL_EXECUTOR.execute(mActive);
            }
        }
    }

    // THREAD_POOL_EXECUTOR就是传说中的线程池，THREAD_POOL_EXECUTOR.execute
    // 会自动使用线程池中空闲的线程完成mActive的任务
    /**
     * An {@link Executor} that can be used to execute tasks in parallel.
     */
    public static final Executor THREAD_POOL_EXECUTOR;

    static {
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
                CORE_POOL_SIZE, MAXIMUM_POOL_SIZE, KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
                sPoolWorkQueue, sThreadFactory);
        threadPoolExecutor.allowCoreThreadTimeOut(true);
        THREAD_POOL_EXECUTOR = threadPoolExecutor;
    }

// 明白了sDefaultExecutor本质上是线程池，接下来看executeOnExecutor怎么调用线程池
    @MainThread
    public final AsyncTask<Params, Progress, Result> executeOnExecutor(Executor exec,
            Params... params) {
        if (mStatus != Status.PENDING) {
            switch (mStatus) {
                case RUNNING:
                    throw new IllegalStateException("Cannot execute task:"
                            + " the task is already running.");
                case FINISHED:
                    throw new IllegalStateException("Cannot execute task:"
                            + " the task has already been executed "
                            + "(a task can be executed only once)");
            }
        }
        // 首先设置了状态为RUNNING
        mStatus = Status.RUNNING;

        // 这里是不是很熟悉，我们继承AsyncTask时重写的onPreExecute方法
        onPreExecute();
        // 然后将参数保存在mWorker.mParams
        mWorker.mParams = params;
        // 调用exec.execute，这里的exec就是上面的SerialExecutor
        exec.execute(mFuture);

        return this;
    }    

// 此时我们需要明白mWorker和mFuture是什么，这里就要了解子类继承父类时，构造方法的执行了
// 在我们MyTask task = new MyTask();时，其实完成了父类的无参构造方法的执行，也就是AsyncTask
// 的无参构造方法
    /**
     * Creates a new asynchronous task. This constructor must be invoked on the UI thread.
     */
    public AsyncTask() {
        this((Looper) null);
    }
    // 也就是说mWorker和mFuture在new MyTask()时已经完成了初始化
    public AsyncTask(@Nullable Looper callbackLooper) {
        // callbackLooper为null，所以mHandler为getMainHandler方法的返回值，
        // 看名字就知道返回的是主线程的Handler，但是这个Handler有点东西
        mHandler = callbackLooper == null || callbackLooper == Looper.getMainLooper()
            ? getMainHandler()
            : new Handler(callbackLooper);

        // mWorker提供了一个对象WorkerRunnable，WorkerRunnable实现了Callable接口的call方法
        // 当mWorker的call方法被执行时，我们就可以得到结果
        mWorker = new WorkerRunnable<Params, Result>() {
            public Result call() throws Exception {
                mTaskInvoked.set(true);
                Result result = null;
                try {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
                    //noinspection unchecked
                    // 很熟悉的doInBackground方法，传入的参数为mWorker的mParams，即我们
                    // 在task.execute("beijing", 50L);传入的参数
                    result = doInBackground(mParams);
                    Binder.flushPendingCommands();
                } catch (Throwable tr) {
                    mCancelled.set(true);
                    throw tr;
                } finally {
                    postResult(result);
                }
                return result;
            }
        };
        // mFuture以mWorker为参数实现了FutureTask，这里的FutureTask可以被SerialExecutor execute，
        // 会调用FutureTask的run方法，run方法中会执行mWorker的call方法，最终会调用FutureTask的done方法
        mFuture = new FutureTask<Result>(mWorker) {
            @Override
            protected void done() {
                try {
                    // get方法得到的是FutureTask执行run方法后得到的result
                    postResultIfNotInvoked(get());
                } catch (InterruptedException e) {
                    android.util.Log.w(LOG_TAG, e);
                } catch (ExecutionException e) {
                    throw new RuntimeException("An error occurred while executing doInBackground()",
                            e.getCause());
                } catch (CancellationException e) {
                    postResultIfNotInvoked(null);
                }
            }
        };
    }

    // postResultIfNotInvoked会进一步处理结果
    private void postResultIfNotInvoked(Result result) {
        final boolean wasTaskInvoked = mTaskInvoked.get();
        if (!wasTaskInvoked) {
            postResult(result);
        }
    }

    // postResult通过主线程的Handler发送了数据result，并标记MESSAGE_POST_RESULT
    // 表示数据已经获取完毕，应该交给主线程处理
    private Result postResult(Result result) {
        @SuppressWarnings("unchecked")
        Message message = getHandler().obtainMessage(MESSAGE_POST_RESULT,
                new AsyncTaskResult<Result>(this, result));
        message.sendToTarget();
        return result;
    }

    // 还是记得上文介绍的Handler吗，它还有额外的功能
    private static class InternalHandler extends Handler {
        public InternalHandler(Looper looper) {
            super(looper);
        }

        @SuppressWarnings({"unchecked", "RawUseOfParameterizedType"})
        @Override
        public void handleMessage(Message msg) {
            // 在处理Message时，还可以判断并选择执行onProgressUpdate
            AsyncTaskResult<?> result = (AsyncTaskResult<?>) msg.obj;
            switch (msg.what) {
                // 上面说的数据获取完毕会标记MESSAGE_POST_RESULT
                case MESSAGE_POST_RESULT:
                    // There is only one result
                    // 调用mTask，这里就是AsyncTask的finish方法
                    result.mTask.finish(result.mData[0]);
                    break;
                case MESSAGE_POST_PROGRESS:
                    result.mTask.onProgressUpdate(result.mData);
                    break;
            }
        }
    }

    // 熟悉的重写isCancelled和onPostExecute
    private void finish(Result result) {
        if (isCancelled()) {
            // 如果主动调用isCancelled则走onCancelled
            onCancelled(result);
        } else {
            // 或者最终回到我们重写的onPostExecute
            onPostExecute(result);
        }
        mStatus = Status.FINISHED;
    }    

// 以上就是正常的AsyncTask执行流程，但是别忘了我们有一个进度显示的功能
    @WorkerThread
    protected final void publishProgress(Progress... values) {
        if (!isCancelled()) {
            // 通过Handler发送进度数据values
            // 这里就对应了上面的主线程的Handler的另一个功能，显示进度
            getHandler().obtainMessage(MESSAGE_POST_PROGRESS,
                    new AsyncTaskResult<Progress>(this, values)).sendToTarget();
        }
    }
```

整理一下流程，很显然的是AsyncTask本质上还是基于Handler，但是在对线程的处理上采用了线程池，具体的执行过程：

1. 在`new MyTask()`时初始化了主线程的Handler和线程池，构造了FutureTask并提供了doInBackground的回调，并提供了通过sendToTarget的方式处理result和progress的方式；
2. 当我们执行`task.execute("beijing", 50L);`的方法时，提供了onPreExecute的回调，并将参数传给第1步中的FutureTask，然后使用SerialExecutor execute第1步构造的的FutureTask，本质上还是线程池，只是附加了功能：连续处理队列中的所有任务；
3. 最后将得到结果通过上面sendToTarget后Handler的回调handleMessage处理发送的数据

仔细思考一下就会发现，AsyncTask提供了一个显示进度的方法，比较适用于上传下载文件的场景，因为下载进度与下载文件的大小是可知的，但是很多http框架比如Retrofit，可以很方便在接受Response的时候监听下载进度，导致AsyncTask无用武之处；同时对于登录注册功能来说，登录进度并不是很适合量化，所以也不适用；还有其他的场景我暂时也没有想到。这就导致了AsyncTask的作用被弱化了，除了集成doInBackground和onPostExecute方法就没有什么亮眼之处。

AsyncTask也提供了带Looper或者Handler参数的构造函数，此时会影响的只有postResult和publishProgress方法，即这两个方法会发送消息到Looper的线程中，但是子线程的Handler需要自定义handleMessage并自行判断msg.what，包括`MESSAGE_POST_RESULT`和`MESSAGE_POST_PROGRESS`，实现从子线程到子线程的消息传递。

## 3. EventBus

EventBus比上面介绍的两种方式更加强大，除了线程间通信之外，还可以在Activity间传递消息，同时兼具灵活的线程切换功能，先直接上一个简单的例子，依然是请求和风天气数据

> 1.首先使用EventBus需要自定义MessageEvent，即通过EventBus传递的消息载体

```java
public class MessageEvent {
    private Object msg;

    public MessageEvent(Object msg) {
        this.msg = msg;
    }

    public Object getMsg() {
        return msg;
    }

    public void setMsg(Object msg) {
        this.msg = msg;
    }

    @Override
    public String toString() {
        return "MessageEvent{" + "msg=" + msg + '}';
    }
}
```

> 2.在需要处理消息的地方（Activity）中定义Subscribe方法，这个方法可以自动接收其他地方传来的消息

```java
// Subscribe注解修饰处理MessageEvent的方法，有几个参数threadMode、sticky、priority
/**
首先是threadMode：
POSTING：默认，表示事件处理函数的线程跟发布事件的线程在同一个线程。

MAIN：表示事件处理函数的线程在主线程(UI)线程，因此在这里不能进行耗时操作。

BACKGROUND：表示事件处理函数的线程在后台线程，因此不能进行UI操作。
如果发布事件的线程是主线程(UI线程)，那么事件处理函数将会开启一个后台线程，
如果果发布事件的线程是在后台线程，那么事件处理函数就使用该线程。

ASYNC：表示无论事件发布的线程是哪一个，事件处理函数始终会新建一个子线程运行，同样不能进行UI操作。

然后是sticky，sticky用于表示是否接收粘性事件

最后是priority，priority决定不同的Subscribe方法接收事件的优先级，数值越大越早接收，
先接受的Subscribe方法还可以禁止事件继续传递下去
1. 只有当两个订阅方法使用相同的ThreadMode参数的时候，它们的优先级才会与priority指定的值一致；
2. 只有当某个订阅方法的ThreadMode参数为POSTING的时候，它才能停止该事件的继续分发。
**/
// requestData方法用于处理发送的消息是String，其他则打印log
@Subscribe(threadMode = ThreadMode.BACKGROUND)
public void requestData(MessageEvent message) {
    if (message.getMsg() instanceof String) {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        Api api = retrofit.create(Api.class);
        Call<WeatherEntity> call = api.getNowWeather(message.getMsg().toString(), KEY);
        try {
            // 通过EventBus把请求得到的天气发送出去
            EventBus.getDefault().post(new MessageEvent(call.execute().body()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    } else if (message.getMsg() instanceof WeatherEntity) {
        Log.i("aaaa", "ThreadMode.BACKGROUND: " + message.getMsg().toString());
    }
}
// showMessage方法用于处理发送的消息是WeatherEntity，将其显示到TextView上，其他类型则打印log
@Subscribe(threadMode = ThreadMode.MAIN)
public void showMessage(MessageEvent message) {
    if (message.getMsg() instanceof WeatherEntity) {
        textView.setText(message.getMsg().toString());
    } else if (message.getMsg() instanceof String) {
        Log.i("aaaa", "ThreadMode.MAIN: " + message.getMsg().toString());
    }
}
```

> 3.在onStart和onStop中注册和取消注册

```java
@Override
protected void onStart() {
    EventBus.getDefault().register(this);
    super.onStart();
}

@Override
protected void onStop() {
    EventBus.getDefault().unregister(this);
    super.onStop();
}
```

> 4.发送消息

```java
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    textView = findViewById(R.id.textView);
    textView.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            // 发送的消息本体是String，即请求的location
            EventBus.getDefault().post(new MessageEvent("beijing"));
        }
    });
}
```

当我们点击textView的时候就会发送消息`new MessageEvent("beijing")`出去，如果打断点观察消息发送的流程就会清楚：

1. 我们在主线程发送的消息会首先由`ThreadMode.MAIN`的方法处理，此时就会打印log：
2. 然后消息会发送到requestData方法，而requestData方法是`ThreadMode.BACKGROUND`，所以它会在子线程中执行，我们在子线程中又发了`new MessageEvent(call.execute().body())`，因此消息会首先由`ThreadMode.BACKGROUND`的方法处理，即requestData方法自身，此时打印log；
3. 最后消息又传到了showMessage方法中，而showMessage方法是`ThreadMode.MAIN`，所以可以执行在主线程，因此textView被设置了text，整个消息传递流程结束。

根据上面的例子基本可以了解了EventBus发送消息的机制，类似于广播，不同的threadMode参数决定这个方法的执行线程，而消息发送时会首先发到当前线程的方法中，如果在这个方法中消息没有被取消，则会继续广播到其他线程的方法中，具体顺序可以测试一下，直到没有可以处理此消息的方法，整个消息传播的流程就结束了。

```java
// 以下列方法处理从主线程发送的MessageEvent，我们观察一下log的顺序
@Subscribe(threadMode = ThreadMode.MAIN)
public void broadcastMessage1(MessageEvent message) {
    Log.i("aaaa", "ThreadMode.MAIN: " + message.getMsg().toString());
}

@Subscribe(threadMode = ThreadMode.BACKGROUND)
public void broadcastMessage2(MessageEvent message) {
    Log.i("aaaa", "ThreadMode.BACKGROUND: " + message.getMsg().toString());
}

@Subscribe(threadMode = ThreadMode.POSTING)
public void broadcastMessage3(MessageEvent message) {
    Log.i("aaaa", "ThreadMode.POSTING: " + message.getMsg().toString());
}

@Subscribe(threadMode = ThreadMode.ASYNC)
public void broadcastMessage4(MessageEvent message) {
    Log.i("aaaa", "ThreadMode.ASYNC: " + message.getMsg().toString());
}
```

依次是MAIN -> POSTING -> BACKGROUND -> ASYNC

```log
2019-07-28 20:40:38.419 28604-28604/com.example.gsondemo I/aaaa: ThreadMode.MAIN: beijing
2019-07-28 20:40:38.420 28604-28604/com.example.gsondemo I/aaaa: ThreadMode.POSTING: beijing
2019-07-28 20:40:38.421 28604-28765/com.example.gsondemo I/aaaa: ThreadMode.BACKGROUND: beijing
2019-07-28 20:40:38.422 28604-28766/com.example.gsondemo I/aaaa: ThreadMode.ASYNC: beijing
```

在看一下从BACKGROUND子线程发送的MessageEvent，依次是BACKGROUND -> POSTING -> MAIN -> ASYNC

```log
2019-07-28 20:53:27.151 30433-30481/com.example.gsondemo I/aaaa: ThreadMode.BACKGROUND: shanghai
2019-07-28 20:53:27.151 30433-30481/com.example.gsondemo I/aaaa: ThreadMode.POSTING: shanghai
2019-07-28 20:53:27.152 30433-30433/com.example.gsondemo I/aaaa: ThreadMode.MAIN: shanghai
2019-07-28 20:53:27.152 30433-30482/com.example.gsondemo I/aaaa: ThreadMode.ASYNC: shanghai
```

消息广播的规则应该是首先是发送到post所在的线程，然后是POSTING，然后是其他线程，最后是ASYNC，因此我们可以在POSTING方法中取消消息的广播，那么消息就会被中断。

* 普通事件删除

```java
EventBus.getDefault().cancelEventDelivery(event);
```

* 粘性事件删除

```java
//指定粘性事件删除  
T stickyEvent = EventBus.getDefault().getStickyEvent(eventType);  
if (stickyEvent != null) {
    EventBus.getDefault().removeStickyEvent(stickyEvent);  
}

//删除所有粘性事件 
EventBus.getDefault().removeAllStickyEvents();
```

除了普通事件之外，EventBus还可以发送粘性事件，解释起来比较复杂，简而言之就是让消息“飞一会”，在我们主动注册时才处理消息，用代码来解释

```java
// 代码是类似的，只是这次不在onStart方法内注册，而是通过button点击注册
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    textView = findViewById(R.id.textView);
    button = findViewById(R.id.button);
    textView.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            // postSticky替代post
            EventBus.getDefault().postSticky(new MessageEvent("beijing"));
        }
    });
    button.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            EventBus.getDefault().register(MainActivity.this);
        }
    });
}

// 增加sticky = true
@Subscribe(threadMode = ThreadMode.BACKGROUND, sticky = true)
public void requestData(MessageEvent message) {
    if (message.getMsg() instanceof String) {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        Api api = retrofit.create(Api.class);
        Call<WeatherEntity> call = api.getNowWeather(message.getMsg().toString(), KEY);
        try {
            EventBus.getDefault().post(new MessageEvent(call.execute().body()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    } else if (message.getMsg() instanceof WeatherEntity) {
        Log.i("aaaa", "ThreadMode.BACKGROUND: " + message.getMsg().toString());
    }
}

// 增加sticky = true
@Subscribe(threadMode = ThreadMode.MAIN, sticky = true)
public void showMessage(MessageEvent message) {
    if (message.getMsg() instanceof WeatherEntity) {
        textView.setText(message.getMsg().toString());
    } else if (message.getMsg() instanceof String) {
        Log.i("aaaa", "ThreadMode.MAIN: " + message.getMsg().toString());
    }
}

@Override
protected void onStart() {
    super.onStart();
}

@Override
protected void onStop() {
    EventBus.getDefault().unregister(this);
    super.onStop();
}
```

这样的结果就是当我们点击textView时，事件就会发出，但是Subscribe方法没有接收，当且仅当我们点击了button时，事件才开始被接收，即我们让消息在运行时“飞了一会”，消息并不会丢失，当我们主动去注册时才开始处理，这就是粘性事件。

priority就不详细解释了，对于有相同threadMode的方法，priority值越大越先接收到消息。

EventBus源码解析暂时留个坑。

## 4. RxJava



```gradle
implementation "io.reactivex.rxjava2:rxjava:2.2.8" // 必要rxjava2依赖
implementation "io.reactivex.rxjava2:rxandroid:2.1.0" // 必要rxandrroid依赖，切线程时需要用到
```

还是以请求和风天气数据为例

```java
Retrofit retrofit = new Retrofit.Builder()
        .baseUrl(URL)
        .addConverterFactory(GsonConverterFactory.create())
        .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
        .build();

Api api = retrofit.create(Api.class);
// 这里的getNowWeather方法在Api.java中返回的是Observable
api.getNowWeather("beijing", KEY)
        .subscribeOn(Schedulers.io()) // subscribeOn参数为io线程，表明getNowWeather请求数据执行在io线程
        .observeOn(AndroidSchedulers.mainThread()) // observeOn参数为主线程，表明请求结束传递的数据在主线程处理
        .subscribe(new Consumer<WeatherEntity>() { // subscribe定义上面observeOn进行的方法，RxJava2中以Consumer代理处理，一般来说有两个Consumer，一个用于处理请求成功的数据，另一个处理异常
            @Override
            public void accept(WeatherEntity weatherEntity) throws Exception {
                textView.setText(weatherEntity.toString());
            }
        }, new Consumer<Throwable>() {
            @Override
            public void accept(Throwable throwable) throws Exception {
                Log.i("aaaa", throwable.getMessage());
            }
        });
```













