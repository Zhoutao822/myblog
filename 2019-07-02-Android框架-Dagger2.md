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
> [Dagger2 最清晰的使用教程](https://www.jianshu.com/p/24af4c102f62?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
> [The New Dagger2](https://blog.mindorks.com/the-new-dagger-2-android-injector-cbe7d55afa6a)

Dagger2框架是一个依赖注入框架，它既可以用于Java Web项目也可以用于Android项目，依赖注入是什么意思呢

```java
public class Dependent {
    private Dependency dependency;

    // 属性注入 
    public Dependent(Dependency dependency) {
        this.dependency = dependency;
    }

    // public Dependent(){
    //     this.dependency = new Dependency();
    // }

    // 方法注入
    // public void setDependency(Dependency dependency){
    //     this.dependency = dependency;
    // }

    private void doSomething(){

    }
}
```

看名字知含义，在上面的代码中Dependent类的构造始终需要Dependency类，那么我们就称Dependency为依赖，将其引入Dependent中的过程称为注入，上述代码在构造函数中引入，当然也可以通过set方法引入，无论是哪种方式都会面临一个问题就是当我们后续如果需要修改Dependency的构造函数时，需要在所有包含`new Dependency()`的代码中进行修改，显然这是非常痛苦的事情，而且不符合依赖倒置原则，本文所涉及到的是通过注解的方式进行依赖注入可以解决这种问题。

<!-- more -->

## 1. Dagger2框架入门

Dagger2框架最终的概念是注解，注解有什么用呢，我觉得是一种标记，这是由于Dagger2框架最终是通过根据不同的注解自动生成代码来实现的依赖注入，因此不同的注解表示通过不同的逻辑生成代码以实现其功能。

从最简单最基础的注解开始，一步一步深入，了解其生成的源码的作用。

### 1.1 @Inject和@Component

比如我们需要一个Utils类

```java
public class Utils {
    public Utils() {
    }

    public String showMessage() {
        return "This is Utils";
    }
}
```

然后在MainActivity中使用showMessage方法

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private Utils utils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 这里需要new一个对象出来才能调用showMessage方法
        utils = new Utils();
        Log.i(TAG, utils.showMessage());
    }
}
```

如果需要在其他Activity中继续使用Utils的showMessage方法，那么就需要重复在每一个Activity中new一个Utils对象，这时候产品经理来了跟你说在使用Utils的时候还需要使用ToastUtils，而且需要修改Utils的构造函数，将ToastUtils传进去

```java
public class ToastUtils {
    public ToastUtils() {
    }

    public String showMessage() {
        return "This is ToastUtils";
    }
}
```

```java
public class Utils {

    private ToastUtils toastUtils;

    public Utils(ToastUtils toastUtils) {
        this.toastUtils = toastUtils;
    }

    public String showMessage() {
        return toastUtils.showMessage();
    }
}
```

此时，你是不是要疯了，需要在所有调用`new Utils()`的位置进行修改，也就意味着每一次修改构造函数都需要全部重新修改一次。

通过dagger2框架是如何实现依赖注入的呢？

* 首先是在依赖的构造函数上加上`@Inject`

```java
public class Utils {
    @Inject
    public Utils() {
    }

    public String showMessage() {
        return "This is Utils";
    }
}
```

* 然后新建一个接口`MainActivityComponent`，要加上`@Component`，声明`inject`方法，参数为依赖被注入的类，这个接口向dagger2框架表明了需要注入的目标，即依赖者dependent

```java
@Component
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

* 最后在MainActivity中使用，直接在依赖上增加注解`@Inject`，在onCreate方法中调用`DaggerMainActivityComponent.create().inject(this);`，然后utils就被实例化了，可以直接使用，这里并没有看见new对象的操作

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    Utils utils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 在调用DaggerMainActivityComponent.create().inject(this)方法前先build一下，
        // 会自动生成一些代码，其中包括DaggerMainActivityComponent类，否则无法使用
        DaggerMainActivityComponent.create().inject(this);

        Log.i(TAG, utils.showMessage());
    }
}
```

我们来看一下生成代码实现了哪些功能吧，主要包括三个类`DaggerMainActivityComponent.java`、`MainActivity_MembersInjector.java`、`Utils_Factory.java`

```java
// DaggerMainActivityComponent.java
// DaggerMainActivityComponent是根据MainActivityComponent生成的，按照执行顺序分析
public final class DaggerMainActivityComponent implements MainActivityComponent {
// 3. DaggerMainActivityComponent构造函数    
  private DaggerMainActivityComponent() {

  }

  public static Builder builder() {
    return new Builder();
  }
// 1. create方法返回Builder().build()方法返回的对象
  public static MainActivityComponent create() {
    return new Builder().build();
  }

// 4. 调用inject方法
  @Override
  public void inject(MainActivity activity) {
    injectMainActivity(activity);}

// 5. inject方法实际执行的方法injectMainActivity
  private MainActivity injectMainActivity(MainActivity instance) {
    // 6. 调用MainActivity_MembersInjector.injectUtils(instance, new Utils())，这里出现了new出来的实例
    // 接下来看MainActivity_MembersInjector类做了些什么
    MainActivity_MembersInjector.injectUtils(instance, new Utils());
    return instance;
  }

  public static final class Builder {
    private Builder() {
    }
// 2. Builder().build()返回的对象是DaggerMainActivityComponent
    public MainActivityComponent build() {
      return new DaggerMainActivityComponent();
    }
  }
}
```

```java
// MainActivity_MembersInjector.java
public final class MainActivity_MembersInjector implements MembersInjector<MainActivity> {
  private final Provider<Utils> utilsProvider;

  public MainActivity_MembersInjector(Provider<Utils> utilsProvider) {
    this.utilsProvider = utilsProvider;
  }

  public static MembersInjector<MainActivity> create(Provider<Utils> utilsProvider) {
    return new MainActivity_MembersInjector(utilsProvider);}

  @Override
  public void injectMembers(MainActivity instance) {
    injectUtils(instance, utilsProvider.get());
  }
// 7. 接上面的执行，这就很明显了instance.utils = utils 等价于 MainActivity.utils = new Utils()
// 也就是说到这里，其实依赖注入的功能就完成了，其他的代码并没有用到，但是不代表是无用的
  public static void injectUtils(MainActivity instance, Utils utils) {
    instance.utils = utils;
  }
}
```

按照增加ToastUtils的方式进行依赖注入是怎样的呢，需要修改如下代码

```java
public class ToastUtils {
    // ToastUtils被Utils依赖，所以需要在构造函数上加上@Inject
    @Inject
    public ToastUtils(){

    }

    public String showMessage(){
        return "This is ToastUtils";
    }
}
```

```java
public class Utils {

    private ToastUtils toastUtils;

    // Utils的含参构造函数上加上@Inject
    @Inject
    public Utils(ToastUtils toastUtils) {
        this.toastUtils = toastUtils;
    }

    public String showMessage() {
        return toastUtils.showMessage();
    }
}
```

`MainActivityComponent.java`和`MainActivity.java`不用修改任何代码，那不就意味着我们解决了前面注入产生的修改代码的问题吗，因为没有new对象的代码；而且ToastUtils在Utils中也不是通过new对象产生的，而是层层注解注入的。

此时再次看一下生成的代码文件`DaggerMainActivityComponent.java`、`MainActivity_MembersInjector.java`、`Utils_Factory.java`、`ToastUtils_Factory.java`

```java
public final class DaggerMainActivityComponent implements MainActivityComponent {
  private DaggerMainActivityComponent() {

  }

  public static Builder builder() {
    return new Builder();
  }

  public static MainActivityComponent create() {
    return new Builder().build();
  }
// getUtils()即返回了我们需要的带参Utils对象
  private Utils getUtils() {
    return new Utils(new ToastUtils());}

  @Override
  public void inject(MainActivity activity) {
    injectMainActivity(activity);}
// 这次直接看核心代码，MainActivity_MembersInjector.injectUtils(instance, getUtils())
// MainActivity_MembersInjector.injectUtils方法也很熟悉了，效果同上文
  private MainActivity injectMainActivity(MainActivity instance) {
    MainActivity_MembersInjector.injectUtils(instance, getUtils());
    return instance;
  }

  public static final class Builder {
    private Builder() {
    }

    public MainActivityComponent build() {
      return new DaggerMainActivityComponent();
    }
  }
}
```

根据上文的分析，我们知道了我们需要的对象的实例其实是在生成的代码`DaggerMainActivityComponent.java`中new出来的，但是这个过程并不需要我们干预而是自动生成的，所以解决了部分依赖注入产生的问题。

结合源码分析可知

1. `@Inject`标注在构造器上的含义包括：

* 告诉Dagger2可以使用这个构造器构建对象。如ToastUtils类
* 注入构造器所需要的参数的依赖。 如Utils类，构造上的ToastUtils会被注入。

构造器注入的局限：如果有多个构造器，我们只能标注其中一个，无法标注多个。

2. `@Component`一般有两种方式定义方法

* void inject(目标类 obj);Dagger2会从目标类开始查找@Inject注解，自动生成依赖注入的代码，调用inject可完成依赖的注入。
* Object getObj(); 如：Utils getUtils();
Dagger2会到Utils类中找被@Inject注解标注的构造器，自动生成提供Utils依赖的代码，这种方式一般为其他Component提供依赖。（一个Component可以依赖另一个Component，后面会说）

使用接口定义，并且`@Component`注解。命名方式推荐为：目标类名+Component，在编译后Dagger2就会为我们生成DaggerXXXComponent这个类，它是我们定义的xxxComponent的实现，在目标类中使用它就可以实现依赖注入了。

### 1.2 @Module和@Provides

使用@Inject标记构造器提供依赖是有局限性的，比如说我们需要注入的对象是第三方库提供的，我们无法在第三方库的构造器上加上@Inject注解。

或者，我们使用依赖倒置的时候，因为需要注入的对象是抽象的，@Inject也无法使用，因为抽象的类并不能实例化，比如：

```java
public abstract class AbstractUtils {

    public abstract String showMessage();
}
```

```java
public class DBUtils extends AbstractUtils {

    @Inject
    DBUtils() {}

    @Override
    public String showMessage() {
        return "This is DBUtils";
    }
}
```

```java
public class ApiUtils extends AbstractUtils {

    @Inject
    ApiUtils() {}

    @Override
    public String showMessage() {
        return "This is ApiUtils";
    }
}
```

```java
public class DataUtils {

    private AbstractUtils abstractUtils;

    @Inject
    public DataUtils(AbstractUtils abstractUtils) {
        this.abstractUtils = abstractUtils;
    }

    public String show() {
        return abstractUtils.showMessage();
    }
}
```

`MainActivityComponent.java`不变，如果在MainActivity中引入DataUtils会报错，此时需要修改代码

```java
public class DBUtils extends AbstractUtils {

    @Override
    public String showMessage() {
        return "This is DBUtils";
    }
}
```

```java
public class ApiUtils extends AbstractUtils {

    @Override
    public String showMessage() {
        return "This is ApiUtils";
    }
}
```

需要新建一个Module类，用于提供需要的实例，这里返回的是DBUtils对象，@Provodes标记在方法上，表示可以通过这个方法获取依赖

```java
@Module
public class DataUtilsModule {

    @Provides
    AbstractUtils provideDataUtils() {
        return new DBUtils();
    }
}
```

修改Component代码

```java
@Component(modules = DataUtilsModule.class)
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后在MainActivity中引入

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();
    @Inject
    DataUtils dataUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        DaggerMainActivityComponent.create().inject(this);
        // 很显然，这里引入的是DBUtils对象
        Log.i(TAG, dataUtils.show());
    }
}
```

通过修改`DataUtilsModule`中`provideDataUtils`方法返回的对象，我们可以控制抽象类的具体子类是DBUtils还是ApiUtils，而主题代码不需要改动。

此时代码分析包括`DaggerMainActivityComponent.java`、`DataUtilsModule_ProvideDataUtilsFactory.java`、``、``、`MainActivity_MembersInjector.java`

```java
public final class DaggerMainActivityComponent implements MainActivityComponent {
  private final DataUtilsModule dataUtilsModule;

  private DaggerMainActivityComponent(DataUtilsModule dataUtilsModuleParam) {
    this.dataUtilsModule = dataUtilsModuleParam;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static MainActivityComponent create() {
    return new Builder().build();
  }
// getDataUtils()返回的是
// new DataUtils(DataUtilsModule_ProvideDataUtilsFactory.provideDataUtils(dataUtilsModule))
// 构造函数的参数为DataUtilsModule_ProvideDataUtilsFactory.provideDataUtils(dataUtilsModule)
// 接下来看这个方法provideDataUtils的返回值
  private DataUtils getDataUtils() {
    return new DataUtils(DataUtilsModule_ProvideDataUtilsFactory.provideDataUtils(dataUtilsModule));}

  @Override
  public void inject(MainActivity activity) {
    injectMainActivity(activity);}
// 直接看核心代码，MainActivity_MembersInjector.injectDataUtils(instance, getDataUtils())
// 根据getDataUtils方法的返回值可知，其返回的是DataUtils实例
// MainActivity_MembersInjector.injectDataUtils方法也是很熟悉，同上
  private MainActivity injectMainActivity(MainActivity instance) {
    MainActivity_MembersInjector.injectDataUtils(instance, getDataUtils());
    return instance;
  }

  public static final class Builder {
    private DataUtilsModule dataUtilsModule;

    private Builder() {
    }

    public Builder dataUtilsModule(DataUtilsModule dataUtilsModule) {
      this.dataUtilsModule = Preconditions.checkNotNull(dataUtilsModule);
      return this;
    }

    public MainActivityComponent build() {
      if (dataUtilsModule == null) {
        this.dataUtilsModule = new DataUtilsModule();
      }
      return new DaggerMainActivityComponent(dataUtilsModule);
    }
  }
}
```

```java
public final class DataUtilsModule_ProvideDataUtilsFactory implements Factory<AbstractUtils> {
  private final DataUtilsModule module;

  public DataUtilsModule_ProvideDataUtilsFactory(DataUtilsModule module) {
    this.module = module;
  }

  @Override
  public AbstractUtils get() {
    return provideDataUtils(module);
  }

  public static DataUtilsModule_ProvideDataUtilsFactory create(DataUtilsModule module) {
    return new DataUtilsModule_ProvideDataUtilsFactory(module);
  }
// 上述代码直接调用的是下面这个方法，返回的是DataUtilsModule.provideDataUtils()
// DataUtilsModule根据我们定义的时候可知，provideDataUtils返回的是new DBUtils()对象
  public static AbstractUtils provideDataUtils(DataUtilsModule instance) {
    return Preconditions.checkNotNull(instance.provideDataUtils(), "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

### 1.3 @Qualifier和@Named








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




