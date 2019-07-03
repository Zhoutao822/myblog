---
title: Android框架-Dagger2
date: 2019-07-02 20:26:19
category:
- Android
tags:
- Android
- Dagger2
- DI
- 依赖注入
---

参考：

> [Android开发之dagger.android--Activity](https://www.jianshu.com/p/2ec39d8b7e98)
> [Dagger](https://dagger.dev/)



<!-- more -->


## dagger.android

dagger框架可以用于Java Web项目同时也可以用于Android项目，但是在Android项目中，最重要最常用的几个组件比如Activity，如果需要进行依赖注入，那会是一个什么样的情形呢。

```java
public class XXXActivity extends AppCompatActivity {

    @Inject
    XXXEntity entity;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        DaggerXXXActivityComponent.create().inject(this);
    }
}
```

```java
@Component(module = XXXEntityModule.class)
public interface XXXActivityComponent {
    void inject(XXXActivity activity);
}
```

```java
@Module
public class XXXEntityModule {
    @Provides
    XXXEntity provideXXXEntity() {
        return new XXXEntity();
    }
}
```

以最简单的单个对象XXXEntity注入，我们需要在每一个Activity中加上`DaggerXXXActivityComponent.create().inject(this);`，每一个XXXActivityComponent又需要指定其module，这样就会产生很多重复的代码，且会引起结构混乱；

有人可能会说，那直接用一个ActivityComponent不行吗，把所有的Activity需要的XXXEntity的module都加进去，那就会产生一个module参数非常长的ActivityComponent，显然这也是不合理的；

还有人说，将那些需要相同XXXEntity的Activity使用相同的XXXActivityComponent，不就可以减少很多代码了，显然，项目的复杂度决定了这样的操作依然会产生很多重复代码。

所以我们的目的是在Activity中使用inject方法时不需要知道是哪个XXXActivityComponent，也就是说用一个通用方法`AndroidInjection.inject(this)`替换`DaggerXXXActivityComponent.create().inject(this)`，这样就可以在BaseActivity中加入这个方法，那么继承自BaseActivity的Activity就不需要再重复写了。

与此同时，如果XXXActivityComponent也能简化或者集成，那就非常完美了，最终我们需要的是自定义XXXEntityModule，用于提供不同Activity需要的注入对象。

那么首先需要回顾一下`DaggerXXXActivityComponent.create().inject(this)`，详情请往上翻，本质上相当于调用`this.XXXEntity = new XXXEntity()`，但是初始化过程Avtivity并不需要知道，都是通过dagger生成的代码执行的结果。

### Injecting Activity objects

官网给出了在Activity中进行依赖注入的步骤，首先过一遍流程，然后再根据代码分析原理：

1. 实现一个Component在自定义Application中注入

```java
// AppComponent.java
// 这里的module参数必须添加AndroidInjectionModule.class，后面的MainActivityModule.class和AppModule.class有其他作用
@Singleton
@Component(modules = {AndroidInjectionModule.class, MainActivityModule.class, AppModule.class})
public interface AppComponent {
    // 这里inject的参数是自定义MyApplication，也说明了这个需要在MyApplication中调用
    void inject(MyApplication application);
}
```

2. 实现一个Subcomponent与需要注入的Activity关联

```java
// MainActivitySubComponent.java
// 这里的module也必须是AndroidInjectionModule.class，且接口继承自AndroidInjector<YourActivity>，
// 同时需要一个Subcomponent.Factory工厂类继承自AndroidInjector.Factory<YourActivity>
// 现在你可能一脸懵逼，这是啥，为什么要这么写，但是没关系，后面肯定会用到
@Subcomponent(modules = AndroidInjectionModule.class)
public interface MainActivitySubComponent extends AndroidInjector<MainActivity> {

    @Subcomponent.Factory
    public interface Factory extends AndroidInjector.Factory<MainActivity> {}
}
```

3. 实现module为你的XXXActivity提供其需要的对象，这一步还有优化的可能，后面介绍

```java
// MainActivityModule.java
// 这里的subcomponents需要上一步定义的MainActivitySubComponent.class，而且这是一个抽象类
@Module(subcomponents = MainActivitySubComponent.class)
public abstract class MainActivityModule {
    // 需要一个这样的抽象工程方法，这个方法还加了很多的注解，暂时不管这些有什么用，但是没有这个方法肯定会报错
    @Binds
    @IntoMap
    @ClassKey(MainActivity.class)
    abstract AndroidInjector.Factory<?>
    bindMainActivityAndroidInjectorFactory(MainActivitySubComponent.Factory factory);

    // 然后需要一个提供对象的provide方法，这个Entity也就是最终我们需要在MainActivity中用到的对象
    // Singleton注解会导致局部单例而不是全局单例，因为只能在MainActivity中使用
    @Provides
    @Singleton
    static Entity provideEntity() {
        return new Entity();
    }
}
```

4. 自定义Application实现HasAndroidInjector接口，并且进行注入

```java
// MyApplication.java
// extends Application implements HasActivityInjector
public class MyApplication extends Application implements HasActivityInjector {

    // 需要DispatchingAndroidInjector对象，并且在activityInjector()方法中返回
    @Inject
    DispatchingAndroidInjector<Activity> dispatchingActivityInjector;

    @Override
    public void onCreate() {
        super.onCreate();
        // 使用第一步定义的Component进行注入
        DaggerAppComponent.create()
                .inject(this);
    }

    @Override
    public AndroidInjector<Activity> activityInjector() {
        return dispatchingActivityInjector;
    }
}
```

5. 最终在Activity中的onCreate方法中调用`AndroidInjection.inject(this)`，在super.onCreate()之前

```java
// MainActivity.java
public class MainActivity extends AppCompatActivity {

    // 首先是Entity对象，它是在MainActivityModule中引入的
    @Inject
    Entity entity;

    // 其次是String对象，它是在AppModule中引入的
    @Inject
    String info;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        AndroidInjection.inject(this);
        
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        TextView textView = findViewById(R.id.text);
        // 这里就可以直接使用entity的方法showMessage()，以及info对象的值
        String text = entity.showMessage() + " - " + info;
        textView.setText(text);
    }
}
```

```java
// Entity.java
// MainActivity中需要的对象，仅作演示
public class Entity {

    private String msg = "Dagger inject";

    public Entity() {
    }

    public String showMessage() {
        return msg;
    }
}
```

```java
// AppModule.java
// AppModule用于提供全局需要的对象，比如Context，或者一些全局设置比如SharedPreferences、数据库名称等等
@Module
public class AppModule {

    // 这里增加了一个String字符，仅作演示
    @Provides
    @Singleton
    String provideGlobalInfo(){
        return "This is global info";
    }
}
```

### Injecting Activity objects源码分析

需要分析源码才能知道问什么上面我们需要定义各种Factory接口以及为什么要在Application中进行注入

那么首先从Activity中开始，这是使用dagger依赖注入的终点，MainActivity中仅有一处与dagger相关`AndroidInjection.inject(this);`

```java
// AndroidInjection.java
  /**
   * Injects {@code activity} if an associated {@link AndroidInjector} implementation can be found,
   * otherwise throws an {@link IllegalArgumentException}.
   *
   * @throws RuntimeException if the {@link Application} doesn't implement {@link
   *     HasActivityInjector}.
   */
  public static void inject(Activity activity) {
    checkNotNull(activity, "activity");
    Application application = activity.getApplication();
    // 这里对application进行了判断，如果没有实现HasActivityInjector，那么会报错
    // 这也是为什么我们自定义的Application需要实现HasActivityInjector接口
    if (!(application instanceof HasActivityInjector)) {
      throw new RuntimeException(
          String.format(
              "%s does not implement %s",
              application.getClass().getCanonicalName(),
              HasActivityInjector.class.getCanonicalName()));
    }
    // 这里调用了application的activityInjector()方法，得到了一个AndroidInjector<Activity>对象
    AndroidInjector<Activity> activityInjector =
        ((HasActivityInjector) application).activityInjector();
    checkNotNull(activityInjector, "%s.activityInjector() returned null", application.getClass());

    // 然后通过AndroidInjector<Activity>对象，调用其inject方法对当前的activity进行注入
    activityInjector.inject(activity);
  }
```

与MainActivity中的`AndroidInjection.inject(this);`相关联的是自定义的MyApplication，且调用了它的`activityInjector()`方法，这也是为什么我们需要在自定义Application中实现`activityInjector()`方法，且返回了一个DispatchingAndroidInjector<Activity>对象

```java
// MyApplication.java
public class MyApplication extends Application implements HasActivityInjector {

    // 根据上文的分析，我们知道了在MainActivity中调用的inject方法其实是调用了dispatchingActivityInjector的inject方法
    // 而这个DispatchingAndroidInjector<Activity>对象竟然也是通过注入的方式获取的，它的来源DaggerAppComponent.create().inject(this);
    // 因此我们需要到AppComponent中找到DispatchingAndroidInjector<Activity>是怎么来的
    @Inject
    DispatchingAndroidInjector<Activity> dispatchingActivityInjector;

    @Override
    public void onCreate() {
        super.onCreate();

        DaggerAppComponent.create().inject(this);
    }

    @Override
    public AndroidInjector<Activity> activityInjector() {
        return dispatchingActivityInjector;
    }
}
```

在分析AppComponent先看看`DaggerAppComponent.create().inject(this);`做了些什么工作，这里代码都比较多，关联了很多其他类，
这里可以按照记号按顺序分析`DaggerAppComponent.create().inject(this)`的调用过程，显然这里有建造者模式

```java
public final class DaggerAppComponent implements AppComponent {
  private Provider<MainActivitySubComponent.Factory> mainActivitySubComponentFactoryProvider;

  private Provider<Entity> provideEntityProvider;

  private Provider<String> provideGlobalInfoProvider;

// 3. Builder().build()返回了DaggerAppComponent对象
  private DaggerAppComponent(AppModule appModuleParam) {

    initialize(appModuleParam);
  }

  public static Builder builder() {
    return new Builder();
  }

// 1. 调用静态方法create，返回了Builder().build()
  public static AppComponent create() {
    return new Builder().build();
  }

  private Map<Class<?>, Provider<AndroidInjector.Factory<?>>>
      getMapOfClassOfAndProviderOfAndroidInjectorFactoryOf() {
    return Collections.<Class<?>, Provider<AndroidInjector.Factory<?>>>singletonMap(
        MainActivity.class, (Provider) mainActivitySubComponentFactoryProvider);
  }

  private DispatchingAndroidInjector<Activity> getDispatchingAndroidInjectorOfActivity() {
    return DispatchingAndroidInjector_Factory.newInstance(
        getMapOfClassOfAndProviderOfAndroidInjectorFactoryOf(),
        Collections.<String, Provider<AndroidInjector.Factory<?>>>emptyMap());
  }

// 4. DaggerAppComponent构造方法里执行了initialize方法，这个initialize对DaggerAppComponent类里面的私有变量进行了初始化
  @SuppressWarnings("unchecked")
  private void initialize(final AppModule appModuleParam) {
    // 4.1 首先是mainActivitySubComponentFactoryProvider，返回了一个Provider对象，
    // 根据注释可以知道Provider用于提供一个已经构造好的用于注入的对象实例，如果调用这个Provider的get方法，
    // 我们就可以得到MainActivitySubComponentFactory对象
    this.mainActivitySubComponentFactoryProvider =
        new Provider<MainActivitySubComponent.Factory>() {
          @Override
          public MainActivitySubComponent.Factory get() {
            return new MainActivitySubComponentFactory();
          }
        };
    // 4.2 provideEntityProvider被赋值为MainActivityModule_ProvideEntityFactory.create()
    // 使用DoubleCheck是因为Entity在provide方法中标注了Singleton，
    // MainActivityModule_ProvideEntityFactory的作用将在下面继续介绍
    this.provideEntityProvider =
        DoubleCheck.provider(MainActivityModule_ProvideEntityFactory.create());
    // 4.3 provideGlobalInfoProvider同理，但是AppModule_ProvideGlobalInfoFactory.create(appModuleParam)
    // 多了一个参数，AppModule_ProvideGlobalInfoFactory的作用将在下面继续介绍
    this.provideGlobalInfoProvider =
        DoubleCheck.provider(AppModule_ProvideGlobalInfoFactory.create(appModuleParam));
  }

  @Override
  public void inject(MyApplication application) {
    injectMyApplication(application);
  }

  private MyApplication injectMyApplication(MyApplication instance) {
    MyApplication_MembersInjector.injectDispatchingActivityInjector(
        instance, getDispatchingAndroidInjectorOfActivity());
    return instance;
  }

  public static final class Builder {
    private AppModule appModule;

    private Builder() {}

    public Builder appModule(AppModule appModule) {
      this.appModule = Preconditions.checkNotNull(appModule);
      return this;
    }

// 2. Builder().build() new了一个AppModule对象，然后返回了DaggerAppComponent(appModule)的对象，
// 还记得AppModule类的功能吗，提供全局对象，其中有一个String provideGlobalInfo()方法
    public AppComponent build() {
      if (appModule == null) {
        this.appModule = new AppModule();
      }
      return new DaggerAppComponent(appModule);
    }
  }

// 5. MainActivitySubComponentFactory类实现了MainActivitySubComponent.Factory接口的create方法，
// 最终还是返回了MainActivitySubComponentImpl对象
  private final class MainActivitySubComponentFactory implements MainActivitySubComponent.Factory {
    @Override
    public MainActivitySubComponent create(MainActivity arg0) {
      Preconditions.checkNotNull(arg0);
      return new MainActivitySubComponentImpl(arg0);
    }
  }

// 6. MainActivitySubComponentImpl对象实现了MainActivitySubComponent接口的inject方法，
// 这是由于MainActivitySubComponent继承自AndroidInjector<MainActivity>
  private final class MainActivitySubComponentImpl implements MainActivitySubComponent {
    private MainActivitySubComponentImpl(MainActivity arg0) {}

    @Override
    public void inject(MainActivity arg0) {
      injectMainActivity(arg0);
    }

// 7. 最终调用inject方法时，我们看到了inject(MainActivity arg0)参数为MainActivity，
// 想必此时你应该猜到了在MainActivity中的一句话AndroidInjection.inject(this)竟然能在异国他乡被实现
    private MainActivity injectMainActivity(MainActivity instance) {
      // 这里的两个方法injectEntity和injectInfo分别对应了我们在MainActivity中注入的两个对象，instance时MainActivity，
      // provideEntityProvider.get()和provideGlobalInfoProvider.get()方法对应上面initialize方法初始化的私有变量，
      // 看这个方法的样子就知道这是对MainActivity进行注入的实际方法，MainActivity_MembersInjector的作用将在下面继续介绍
      MainActivity_MembersInjector.injectEntity(
          instance, DaggerAppComponent.this.provideEntityProvider.get());
      MainActivity_MembersInjector.injectInfo(
          instance, DaggerAppComponent.this.provideGlobalInfoProvider.get());
      return instance;
    }
  }
}

```




