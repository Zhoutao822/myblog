---
title: Android IPC-AIDL、Messenger和Socket
date: 2019-07-25 21:41:57
category:
- Android
tags:
- Android
- IPC
- AIDL
- Binder
- Messenger
- Socket
- 进程间通信
---

参考：

> [Android开发艺术探索](https://item.m.jd.com/product/11760209.html)

Android系统中的每一个应用都运行在一个各自的进程中，那么不同的应用是如何进行数据交互的呢，大致分为两类，第一类我称之为伪进程间通信，其特征是不同进程都对同一个文件进行操作，数据交互通过此文件，比如两个进程共同读写同一个数据库；第二类我称之为真进程间通信，特征是基于系统级别的Binder进行服务调用从而实现的进程间通信或者Socket，具体细节后面再说。

<!-- more -->

## 1. Socket

首先了解一下Socket，Socket是对TCP和UDP协议的封装，通过Socket建立的连接可以实现互联网中任意两个进程间的通信，不仅限于局域网或者单机多进程，而我们仅需要确定的是设备的IP和监听的端口号，下面看看Socket是如何实现局域网内从手机传数据到笔记本上。

Socket的使用分为服务端和客户端，服务端需要监听自己设定的端口，客户端需要知道服务端的IP和服务端监听的端口，两者通过Socket建立连接，然后以数据流的形式通过Socket传输数据。

首先是服务端，在笔记本上运行的代码，同时需要知道笔记本的IP地址（笔记本和手机在同一局域网中）

```java
// 自定义线程Server
public class Server extends Thread {
    ServerSocket server = null;

    public Server(int port) {
        // Server线程初始化需要指定端口号，且服务端使用ServerSocket建立连接
        try {
            server = new ServerSocket(port);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {
        super.run();
        try {
            System.out.println("服务器在启动中...等待用户的连接");
            // 一直接收用户的连接，连接之后发送一条短信给用户
            while (true) {
                // 建立socket接口，accept方法是一个阻塞进程,等到有用户连接才往下走
                // 定义Socket类
                Socket socket = server.accept();
                // 通过socket对象可以获得输出流，用来写数据
                OutputStream os = socket.getOutputStream();
                // 向客户端发送消息
                os.write("服务器正在向你发送消息！".getBytes());
                // 在服务器上显示连接的上的电脑、
                System.out.println(socket.getInetAddress().getHostAddress() + "连接上了！");
                // 通过socket对象可以获得输入流，用来读取用户数据
                InputStream is = socket.getInputStream();
                // 读取数据
                int len = 0;
                byte[] buf = new byte[1024];
                while ((len = is.read(buf)) != -1) {
                    // 直接把获得的数据打印出来
                    String msgFromClient = new String(buf, 0, len);
                    System.out.println("服务器接收到客户端的数据：" + msgFromClient);
                    // 根据客户端传来的数据，我们再返回数据给客户端
                    if (msgFromClient.contains("hello")) {
                        os.write("hello too!!".getBytes());
                    } else if (msgFromClient.contains("are")) {
                        os.write("I'm fine.".getBytes());
                    } else if (msgFromClient.contains("bye")) {
                        os.write("Bye!".getBytes());
                    }
                }
                os.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

```java
// 在main函数中启用即可，这里选择监听的端口为6768
public class ServerTest {
    public static void main(String[] args) {
        // 这里服务器只需要定义一个端口号就可以了，程序会自动获取IP地址
        // 但是客户端需要连接这个服务器时，需要知道它的IP地址还有端口号
        // ip地址的查看方法：进入cmd窗口，输入ipconfig/all可以看到
        Server server = new Server(6768);
        server.start();
    }
}
```

然后是客户端的设计，需要权限`<uses-permission android:name="android.permission.INTERNET" />`

> 1.简单布局

```java
public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Socket socket = null;

    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button = findViewById(R.id.button);
        button.setOnClickListener(this);
        Button button1 = findViewById(R.id.hello);
        button1.setOnClickListener(this);
        Button button2 = findViewById(R.id.how);
        button2.setOnClickListener(this);
        Button button3 = findViewById(R.id.bye);
        button3.setOnClickListener(this);
        textView = findViewById(R.id.textView);
    }
}    
```

> 2.定义连接Socket的方法

```java
private void connectToServer(String host, int port) {
    // 需要的参数是IP和端口号，就是局域网中你的笔记本的IP和监听的端口号，对应上文的6768
    while (socket == null) {
        try {
            // 建立连接
            socket = new Socket(host, port);
            // 读socket里面的数据
            InputStream s = socket.getInputStream();
            final byte[] buf = new byte[1024];
            int len = 0;
            while ((len = s.read(buf)) != -1) {
                final int finalLen = len;
                // 在主线程中更新读取的数据
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        textView.setText(new String(buf, 0, finalLen));
                    }
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

> 3.创建发送数据到服务器的线程

```java
class SendMessThread extends Thread {

    private String message;

    public SendMessThread(String message) {
        this.message = message;
    }

    @Override
    public void run() {
        super.run();
        //写操作
        if (socket != null) {
            try {
                // 向Socket中的OutputStream写数据即可
                OutputStream os = socket.getOutputStream();
                os.write(("客户端:" + message).getBytes());
                os.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

> 4.最后是点击事件的处理

```java
@Override
public void onClick(View v) {
    switch (v.getId()) {
        case R.id.button:
            new Thread(new Runnable() {
                @Override
                public void run() {
                    // 我笔记本的IP为192.168.31.43
                    connectToServer("192.168.31.43", 6768);
                }
            }).start();
            break;
        case R.id.hello:
            new SendMessThread("hello").start();
            break;
        case R.id.how:
            new SendMessThread("how are you?").start();
            break;
        case R.id.bye:
            new SendMessThread("bye").start();
            break;
    }
}
```

运行结果如下：

> 服务器（笔记本）

```log
服务器在启动中...等待用户的连接
192.168.31.100连接上了！
服务器接收到客户端的数据：客户端:hello
服务器接收到客户端的数据：客户端:how are you?
服务器接收到客户端的数据：客户端:bye
```

> 客户端（手机）

{% asset_img socket.gif %}

这样就完成了服务器与客户端的对话。

## 2. AIDL



## 3. 

## 4. 








