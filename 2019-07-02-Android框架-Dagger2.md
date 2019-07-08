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

看名字知含义，在上面的代码中Dependent类的构造始终需要Dependency类，那么我们就称Dependency为依赖，将其引入Dependent中的过程称为注入，上述代码在构造函数中引入，当然也可以通过set方法注入，无论是哪种方式都会面临一个问题就是当我们后续如果需要修改Dependency的构造函数时，需要在所有包含`new Dependency()`的代码中进行修改，显然这是非常痛苦的事情，而且不符合依赖倒置原则，本文所涉及到的是通过注解的方式进行依赖注入可以解决这种问题。

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
public class AbstractUtilsModule {

    @Provides
    AbstractUtils provideDataUtils() {
        return new DBUtils();
    }
}
```

修改Component代码

```java
@Component(modules = AbstractUtilsModule.class)
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

通过修改`AbstractUtilsModule`中`provideDataUtils`方法返回的对象，我们可以控制抽象类的具体子类是DBUtils还是ApiUtils，而主体代码不需要改动。

此时代码分析包括`DaggerMainActivityComponent.java`、`AbstractUtilsModule_ProvideDataUtilsFactory.java`、`MainActivity_MembersInjector.java`

```java
public final class DaggerMainActivityComponent implements MainActivityComponent {
  private final AbstractUtilsModule AbstractUtilsModule;

  private DaggerMainActivityComponent(AbstractUtilsModule abstractUtilsModuleParam) {
    this.AbstractUtilsModule = abstractUtilsModuleParam;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static MainActivityComponent create() {
    return new Builder().build();
  }
// 2. getDataUtils()返回的是
// new DataUtils(AbstractUtilsModule_ProvideDataUtilsFactory.provideDataUtils(AbstractUtilsModule))
// 构造函数的参数为AbstractUtilsModule_ProvideDataUtilsFactory.provideDataUtils(AbstractUtilsModule)
// 接下来看这个方法provideDataUtils的返回值
  private DataUtils getDataUtils() {
    return new DataUtils(AbstractUtilsModule_ProvideDataUtilsFactory.provideDataUtils(AbstractUtilsModule));}

  @Override
  public void inject(MainActivity activity) {
    injectMainActivity(activity);}
// 1. 直接看核心代码，MainActivity_MembersInjector.injectDataUtils(instance, getDataUtils())
// 根据getDataUtils方法的返回值可知，其返回的是DataUtils实例
// MainActivity_MembersInjector.injectDataUtils方法也是很熟悉，同上
  private MainActivity injectMainActivity(MainActivity instance) {
    MainActivity_MembersInjector.injectDataUtils(instance, getDataUtils());
    return instance;
  }

  public static final class Builder {
    private AbstractUtilsModule AbstractUtilsModule;

    private Builder() {
    }

    public Builder AbstractUtilsModule(AbstractUtilsModule AbstractUtilsModule) {
      this.AbstractUtilsModule = Preconditions.checkNotNull(AbstractUtilsModule);
      return this;
    }

    public MainActivityComponent build() {
      if (AbstractUtilsModule == null) {
        this.AbstractUtilsModule = new AbstractUtilsModule();
      }
      return new DaggerMainActivityComponent(AbstractUtilsModule);
    }
  }
}
```

```java
public final class AbstractUtilsModule_ProvideDataUtilsFactory implements Factory<AbstractUtils> {
  private final AbstractUtilsModule module;

  public AbstractUtilsModule_ProvideDataUtilsFactory(AbstractUtilsModule module) {
    this.module = module;
  }

  @Override
  public AbstractUtils get() {
    return provideDataUtils(module);
  }

  public static AbstractUtilsModule_ProvideDataUtilsFactory create(AbstractUtilsModule module) {
    return new AbstractUtilsModule_ProvideDataUtilsFactory(module);
  }
// 3. 上述代码直接调用的是下面这个方法，返回的是AbstractUtilsModule.provideDataUtils()
// AbstractUtilsModule根据我们定义的时候可知，provideDataUtils返回的是new DBUtils()对象
  public static AbstractUtils provideDataUtils(AbstractUtilsModule instance) {
    return Preconditions.checkNotNull(instance.provideDataUtils(), "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

`@Module`的含义是 通知Component，可以从我这里获取到构造好的对象；

`@Provide`通常是在标记了`@Module`的类中用于标记返回实例的方法，根据我们的使用以及代码分析来看，实例的注入是根据类型自动判断的，也就是说，从MainActivity到Module的实例传递过程中，同一Module中同一类型的provide方法只能存在一个，否则就会报错，比如我们如果在AbstractUtilsModule中再加入一个provideDataUtils2方法，同样返回类型为AbstractUtils，那么MainActivity中的dataUtils就会遇到`依赖迷失`的问题，这两个方法返回一样，那该用哪一个，于是报错，此时可以通过`限定符`，也就是下文介绍的`@Qualifier`和`@Named`来区分。

### 1.3 @Qualifier和@Named

直接上代码，首先是AbstractUtilsModule，通过添加`@Named`并指定一个字符来区别不同的实例，注意这里的方法名其实作用不大，完全依靠限定符区分，这里两个provide方法分别返回之前的两个AbstractUtils的子类DBUtils和ApiUtils。

```java
@Module
public class AbstractUtilsModule {

    @Provides
    @Named("DBUtils")
    AbstractUtils provideDBUtils() {
        return new DBUtils();
    }


    @Provides
    @Named("ApiUtils")
    AbstractUtils provideApiUtils() {
        return new ApiUtils();
    }
}
```

同时，需要修改DataUtils类，因为AbstractUtilsModule表示我们可以提供两种AbstractUtils，你到底要哪个的实例，此时需要在DataUtils构造函数的参数中加上`@Named`注解，与上面对应，表示我需要哪一种AbstractUtils

```java
public class DataUtils {

    private AbstractUtils abstractUtils;

    @Inject
    public DataUtils(@Named("ApiUtils") AbstractUtils abstractUtils) {
        this.abstractUtils = abstractUtils;
    }

    public String show() {
        return abstractUtils.showMessage();
    }
}
```

其他代码不用修改，此时MainActivity中`dataUtils.show()`自然用的是ApiUtils。

`@Qualifier`的作用与`@Named`的作用差不多，但是不需要自定义字符串，使用`@Qualifier`时不是直接用，而是通过`@Qualifier`自定义限定符

```java
@Qualifier
@Retention(RetentionPolicy.RUNTIME)
public @interface DBDataUtils {
}
```

```java
@Qualifier
@Retention(RetentionPolicy.RUNTIME)
public @interface ApiDataUtils {
}
```

`RetentionPolicy.RUNTIME`代表注解会保存在.class文件，虚拟机会在运行时保留，具体有什么区别，暂时还不清楚。

需要修改的地方有两处，一是AbstractUtilsModule，二是DataUtils，都是将`@Named`注解修改为自定义的限定符注解，此时Log结果变为DBUtils

```java
@Module
public class AbstractUtilsModule {

    @Provides
    @DBDataUtils
    AbstractUtils provideDBUtils() {
        return new DBUtils();
    }


    @Provides
    @ApiDataUtils
    AbstractUtils provideApiUtils() {
        return new ApiUtils();
    }
}
```

```java
public class DataUtils {

    private AbstractUtils abstractUtils;

    @Inject
    public DataUtils(@DBDataUtils AbstractUtils abstractUtils) {
        this.abstractUtils = abstractUtils;
    }

    public String show() {
        return abstractUtils.showMessage();
    }
}
```

在上面的代码中我们对DataUtils的构造方法进行Inject注解，这样的操作不是很合适，因为需要尽量少对实体类进行额外的修改，所以我们同样可以通过Module的方式provide一个DataUtils的对象，并在Module中对DataUtils的构造进行约束

```java
// 删掉Inject注解和限定符注解
public class DataUtils {

    private AbstractUtils abstractUtils;

    public DataUtils(AbstractUtils abstractUtils) {
        this.abstractUtils = abstractUtils;
    }

    public String show() {
        return abstractUtils.showMessage();
    }
}
```

新建一个DataUtilsModule用于提供DataUtils对象

```java
@Module
public class DataUtilsModule {

    @Provides
    DataUtils provideDataUtils(@ApiDataUtils AbstractUtils abstractUtils) {
        return new DataUtils(abstractUtils);
    }
}
```

修改MainActivityComponent的modules参数，增加DataUtilsModule.class

```java
@Component(modules = {AbstractUtilsModule.class, DataUtilsModule.class})
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后编译运行，log结果是ApiUtils，根据上述代码的运行，我们能够总结出一些运行的规则：

1. 首先当我们在MainActivity中调用inject方法时，其实是通过MainActivityComponent询问，可以去哪里找到我们需要的实例；
2. MainActivityComponent定义了`modules = {AbstractUtilsModule.class, DataUtilsModule.class}`，告诉系统去这两个类中找；
3. AbstractUtilsModule告诉系统我这里只能提供限定类型的AbstractUtils，DataUtilsModule告诉系统我这里可以提供DataUtils，这各类型恰好与MainActivity需要的实例类型相同，于是系统到DataUtilsModule的provide方法去找；
4. 此时系统发现DataUtilsModule的provide方法是带参数AbstractUtils的，而且还有限定符，那么同样需要能够提供AbstractUtils的Module，这不是恰好与第3步的AbstractUtilsModule相同吗，那么就按照参数限定符找到AbstractUtilsModule的对应provide方法，结果发现它可以直接返回new ApiUtils()对象，正合我心，且不需要继续走下去，那么实例化完成。

在寻找实例的路线中需要用到的Module都必须加在Component中的modules参数中，否则这条路线走不通（DataUtilsModule -> AbstractUtilsModule），存在的问题是你需要知道所有路线上的Module并且将其加入到Component中，显然对于多级依赖产生的多个Module这是不合适的。


### 1.4 @Component的dependence和@SubComponent

Component除了可以提供inject方法以外还可以像Module一样提供实例，这样便于解决多级依赖导致的Module增加问题。

首先创建提供ApiUtils和DBUtils实例的Component，其`modules = AbstractUtilsModule.class`，表明最终方法获取实例还是从Module拿到的，我Component只是交接一下，向外提供接口getDBUtils和getApiUtils

```java
// 同理这里的两个方法也需要限定符注解，表明需要从AbstractUtilsModule拿哪种实例，一般命名get+XXXEntity
@Component(modules = AbstractUtilsModule.class)
public interface AbstractUtilsComponent {

    @DBDataUtils
    AbstractUtils getDBUtils();

    @ApiDataUtils
    AbstractUtils getApiUtils();

}
```

然后我们的MainActivity是需要DataUtils的实例，那么我们也需要提供DataUtils的Component

```java
// modules = DataUtilsModule.class表明实例来源于DataUtilsModule
// dependencies = AbstractUtilsComponent.class表明我们可能需要AbstractUtilsComponent提供的实例
// 逻辑上也是对应的DataUtilsModule返回实例需要限定AbstractUtils，AbstractUtilsComponent恰好可以提供
// 根据1.3的最后部分我们知道这里的dependencies = AbstractUtilsComponent.class可以替换为AbstractUtilsModule，
// 但是这样会失去依赖的层层分离的特点
@Component(modules = DataUtilsModule.class, dependencies = AbstractUtilsComponent.class)
public interface DataUtilsComponent {
    DataUtils getDataUtils();
}
```

其次需要修改MainActivityComponent的参数，这里可以发现不再使用modules参数二是dependencies

```java
// dependencies = DataUtilsComponent.class表明可能需要DataUtilsComponent提供的实例，
// 通过上面定义的getDataUtils方法得到
@Component(dependencies = DataUtilsComponent.class)
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后需要修改MainActivity调用inject的流程，其他代码不变

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    DataUtils dataUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 可以发现调用流程变复杂了，而且恰好在Component加入了dependencies的位置
        // 需要调用dataUtilsComponent或者abstractUtilsComponent来进行初始化，
        // 有人会说了，这不是把流程变复杂了吗，之前都不需要额外的参数，现在需要将生成的Component带入
        // 到初始化的过程，非也非也，此处可以使用DaggerDataUtilsComponent，那当然可以使用继承自DataUtilsComponent
        // 的其他Component，也就是说我们增加了注入的依赖范围，变为可以动态修改的了，注入过程更加灵活
        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(DaggerDataUtilsComponent
                        .builder()
                        .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                        .build())
                .build()
                .inject(this);

        // 很显然，这里引入的是ApiUtils对象
        Log.i(TAG, dataUtils.show());
    }
}
```

下面展示一下为什么说注入的方式变得灵活了，产品经理突然告诉你需要新增一个Utils叫做ExtraUtils，在MainActivity中需要使用

```java
public class ExtraUtils extends AbstractUtils {
    @Override
    public String showMessage() {
        return "This is ExtraUtils";
    }
}
```

在不修改上述的大部分代码的条件下，如何将ExtraUtils注入，首先新建一个ExtraDataUtilsModule用于提供通过ExtraUtils构造的DataUtils

```java
// 这里没有用带参的构造方法，也没有用抽象类，直接实例化，便于演示
@Module
public class ExtraDataUtilsModule{
  
    @Provides
    DataUtils provideDataUtils() {
        return new DataUtils(new ExtraUtils());
    }
}
```

然后新建一个ExtraDataUtilsComponent用于提供DataUtils实例，是不是和DataUtilsComponent功能很像，没错，这里可以用继承，并且由于MainActivityComponent的`dependencies = DataUtilsComponent.class`不变，我用继承来实现即可

```java
// 这里的modules = ExtraDataUtilsModule.class变成了我们自己新定义的Module，它是直接返回new DataUtils(new ExtraUtils())，
// 所以实际上后面的dependencies = AbstractUtilsComponent.class不需要，但是这里保留是为了减少修改MainActivity的代码，当然也可以去掉
@Component(modules = ExtraDataUtilsModule.class, dependencies = AbstractUtilsComponent.class)
public interface ExtraDataUtilsComponent extends DataUtilsComponent{
}
```

最后修改MainActivity中的inject流程

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    DataUtils dataUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 注意这里是DaggerExtraDataUtilsComponent，即新创建的Component
        // 而且abstractUtilsComponent是可以去掉的，同理ExtraDataUtilsComponent的dependencies也需要去掉
        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(DaggerExtraDataUtilsComponent
                        .builder()
                        .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                        .build())
                .build()
                .inject(this);

        // 很显然，这里引入的是ExtraUtils对象
        Log.i(TAG, dataUtils.show());
    }
}
```

通过这样的方式进行额外的依赖注入，就可以避免大部分代码的重构，而仅仅是增加代码，且注入的方式变得灵活

从源码中也可以看到dependencies的作用，以MainActivityComponent为代表

```java
public final class DaggerMainActivityComponent implements MainActivityComponent {
  private final DataUtilsComponent dataUtilsComponent;

  private DaggerMainActivityComponent(DataUtilsComponent dataUtilsComponentParam) {
    this.dataUtilsComponent = dataUtilsComponentParam;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public void inject(MainActivity activity) {
    injectMainActivity(activity);
  }

// 我们可以调用DataUtilsComponent的getDataUtils()方法了
  private MainActivity injectMainActivity(MainActivity instance) {
    MainActivity_MembersInjector.injectDataUtils(
        instance,
        Preconditions.checkNotNull(
            dataUtilsComponent.getDataUtils(),
            "Cannot return null from a non-@Nullable component method"));
    return instance;
  }

  public static final class Builder {
    private DataUtilsComponent dataUtilsComponent;

    private Builder() {}
// 此处为区别，初始化的过程中增加了dataUtilsComponent方法，用于引入DataUtilsComponent
    public Builder dataUtilsComponent(DataUtilsComponent dataUtilsComponent) {
      this.dataUtilsComponent = Preconditions.checkNotNull(dataUtilsComponent);
      return this;
    }

    public MainActivityComponent build() {
      Preconditions.checkBuilderRequirement(dataUtilsComponent, DataUtilsComponent.class);
      return new DaggerMainActivityComponent(dataUtilsComponent);
    }
  }
}
```

以上就是`@Component`的部分使用，包括dependencies参数的意义，与之相似的有`@Subcomponent`注解，让我们回到加入ExtraUtils之前的场景，用`@Subcomponent`实现Componet的的层层依赖。

首先修改AbstractUtilsComponent，增加它的上一级Component的plus方法，参数为上一级Component的Module

```java
@Component(modules = AbstractUtilsModule.class)
public interface AbstractUtilsComponent {

//    @DBDataUtils
//    AbstractUtils getDBUtils();
//
//    @ApiDataUtils
//    AbstractUtils getApiUtils();

    DataUtilsComponent plus(DataUtilsModule dataUtilsModule);

}
```

然后修改DataUtilsComponent，同理，但是修改注解为`@Subcomponent`，且删除了dependencies参数

```java
@Subcomponent(modules = DataUtilsModule.class)
public interface DataUtilsComponent {
    //    DataUtils getDataUtils();
    MainActivityComponent plus();
}
```

其次是MainActivityComponent，同理，修改注解为`@Subcomponent`，且删除了dependencies参数

```java
@Subcomponent()
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后修改MainActivity

```java
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    DataUtils dataUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 此处的注入调用的流程发生了很大的变化
        // 之前是从MainActivityComponent -> DataUtilsComponent -> AbstractUtilsComponent
        // 现在是AbstractUtilsComponent -> DataUtilsComponent -> MainActivityComponent
        // 并且传入的参数变成了Module，这里也增加了注入的灵活性
        DaggerAbstractUtilsComponent
                .create()
                .plus(new DataUtilsModule())
                .plus()
                .inject(this);

        // 很显然，这里引入的是ApiUtils对象
        Log.i(TAG, dataUtils.show());
    }
}
```

从生成的代码不难发现，这次只有一个Component生成文件`DaggerAbstractUtilsComponent.java`，我们看看为什么会这样

```java
// 还是按照执行顺序分析
public final class DaggerAbstractUtilsComponent implements AbstractUtilsComponent {
  private final AbstractUtilsModule abstractUtilsModule;
// 3. 构造函数初始化完成
  private DaggerAbstractUtilsComponent(AbstractUtilsModule abstractUtilsModuleParam) {
    this.abstractUtilsModule = abstractUtilsModuleParam;
  }

  public static Builder builder() {
    return new Builder();
  }
// 1. 返回Builder().build()
  public static AbstractUtilsComponent create() {
    return new Builder().build();
  }
// 4. 执行plus(new DataUtilsModule())方法，返回DataUtilsComponentImpl(dataUtilsModule)
  @Override
  public DataUtilsComponent plus(DataUtilsModule dataUtilsModule) {
    Preconditions.checkNotNull(dataUtilsModule);
    return new DataUtilsComponentImpl(dataUtilsModule);
  }

  public static final class Builder {
    private AbstractUtilsModule abstractUtilsModule;

    private Builder() {}

    public Builder abstractUtilsModule(AbstractUtilsModule abstractUtilsModule) {
      this.abstractUtilsModule = Preconditions.checkNotNull(abstractUtilsModule);
      return this;
    }
// 2. 初始化了abstractUtilsModule为AbstractUtilsModule，这肯定是AbstractUtilsComponent指定了modules参数
// 导致，然后返回DaggerAbstractUtilsComponent的构造函数
    public AbstractUtilsComponent build() {
      if (abstractUtilsModule == null) {
        this.abstractUtilsModule = new AbstractUtilsModule();
      }
      return new DaggerAbstractUtilsComponent(abstractUtilsModule);
    }
  }

  private final class DataUtilsComponentImpl implements DataUtilsComponent {
    private final DataUtilsModule dataUtilsModule;
// 5. 初始化DataUtilsComponentImpl
    private DataUtilsComponentImpl(DataUtilsModule dataUtilsModuleParam) {
      this.dataUtilsModule = dataUtilsModuleParam;
    }
// 10. DataUtilsModule_ProvideDataUtilsFactory.provideDataUtils，接下来看DataUtilsModule_ProvideDataUtilsFactory
// 和AbstractUtilsModule_ProvideApiUtilsFactory
    private DataUtils getDataUtils() {
      return DataUtilsModule_ProvideDataUtilsFactory.provideDataUtils(
          dataUtilsModule,
          AbstractUtilsModule_ProvideApiUtilsFactory.provideApiUtils(
              DaggerAbstractUtilsComponent.this.abstractUtilsModule));
    }
// 6. 调用plus()方法，返回的是MainActivityComponentImpl
    @Override
    public MainActivityComponent plus() {
      return new MainActivityComponentImpl();
    }

    private final class MainActivityComponentImpl implements MainActivityComponent {
// 7. MainActivityComponentImpl()构造
      private MainActivityComponentImpl() {}
// 8. 调用inject方法
      @Override
      public void inject(MainActivity activity) {
        injectMainActivity(activity);
      }
// 9. 最终调用的位置injectDataUtils，这个很熟悉了，instance.dataUtils = DataUtilsComponentImpl.this.getDataUtils()
// getDataUtils返回值是什么呢，看10
      private MainActivity injectMainActivity(MainActivity instance) {
        MainActivity_MembersInjector.injectDataUtils(
            instance, DataUtilsComponentImpl.this.getDataUtils());
        return instance;
      }
    }
  }
}
```

```java
public final class DataUtilsModule_ProvideDataUtilsFactory implements Factory<DataUtils> {
  private final DataUtilsModule module;

  private final Provider<AbstractUtils> abstractUtilsProvider;

  public DataUtilsModule_ProvideDataUtilsFactory(
      DataUtilsModule module, Provider<AbstractUtils> abstractUtilsProvider) {
    this.module = module;
    this.abstractUtilsProvider = abstractUtilsProvider;
  }

  @Override
  public DataUtils get() {
    return provideDataUtils(module, abstractUtilsProvider.get());
  }

  public static DataUtilsModule_ProvideDataUtilsFactory create(
      DataUtilsModule module, Provider<AbstractUtils> abstractUtilsProvider) {
    return new DataUtilsModule_ProvideDataUtilsFactory(module, abstractUtilsProvider);
  }
// 11-1. 接上文10，instance.provideDataUtils即我们定义的DataUtilsModule.provideDataUtils，返回DataUtils实例
  public static DataUtils provideDataUtils(DataUtilsModule instance, AbstractUtils abstractUtils) {
    return Preconditions.checkNotNull(
        instance.provideDataUtils(abstractUtils),
        "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

```java
public final class AbstractUtilsModule_ProvideApiUtilsFactory implements Factory<AbstractUtils> {
  private final AbstractUtilsModule module;

  public AbstractUtilsModule_ProvideApiUtilsFactory(AbstractUtilsModule module) {
    this.module = module;
  }

  @Override
  public AbstractUtils get() {
    return provideApiUtils(module);
  }

  public static AbstractUtilsModule_ProvideApiUtilsFactory create(AbstractUtilsModule module) {
    return new AbstractUtilsModule_ProvideApiUtilsFactory(module);
  }
// 11-2. 接上文10，instance.provideApiUtils即我们定义的AbstractUtilsModule.provideApiUtils，返回ApiUtils实例
  public static AbstractUtils provideApiUtils(AbstractUtilsModule instance) {
    return Preconditions.checkNotNull(
        instance.provideApiUtils(), "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

分析两种调用方式的逻辑可以知道：

1. dependencies方式从最上级的Component一级一级往下调用，获取需要的实例；Subcomponent方式最下级的Component通过一步一步构造出上级Component来调用，在每一步的plus方法中中加入Module提供需要的实例；
2. dependencies方式适用于需要在某个类中注入非常多的其他实例，通过dependencies参数加深；Subcomponent方式适用于将某一个实例提供给其他实例注入，比如将Application context给其他例如ToastUtils、SharedpreferencesUtils使用，Application context作为Component，其他作为Subcomponent；
3. Component dependencies 能单独使用，而Subcomponent必须由Component调用方法获取；
4. Component dependencies 可以很清楚的得知他依赖哪个Component， 而Subcomponent不知道它自己的谁的孩子。

**Component dependencies和Subcomponent使用上的总结**

Component Dependencies：

1. 你想保留独立的想个组件（Flower可以单独使用注入，Pot也可以）
2. 要明确的显示该组件所使用的其他依赖

Subcomponent：

1. 两个组件之间的关系紧密
2. 你只关心Component，而Subcomponent只是作为Component的拓展，可以通过Component.xxx调用。

### 1.5 @Scope和@Singleton

`@Scope`是用来管理依赖的生命周期的。它和`@Qualifier`一样是用来自定义注解的，而`@Singleton`则是`@Scope`的默认实现。

在没有引入`@Scope`时，我们在MainActivity中初始化另一个DataUtils会是什么情况，这两个DataUtils会是相同的吗

```java
public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();
// 这里直接inject第二个dataUtils2
    @Inject
    DataUtils dataUtils;

    @Inject
    DataUtils dataUtils2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TextView textView = findViewById(R.id.text);

        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(DaggerDataUtilsComponent
                        .builder()
                        .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                        .build())
                .build()
                .inject(this);

        // 然后在Log中打印这两个DataUtils对象，结果是这两个对象是不同的，相当于重新new一个
        Log.i(TAG, dataUtils.toString() + dataUtils2.toString());

        textView.setText(dataUtils.show());
    }
}
```

一般而言，我们希望这种工具类Utils是单例模式，比如读写数据库的时候，如果不是单例，那么可能存在“读后写”等问题导致数据不同步，那么单例模式，特别是全局单例就显得非常好用了。

怎样在dagger框架中使用单例，很显然，必定是通过注解来实现`@Singleton`，在上面的代码中，我们需要的单例是DataUtils，那么从DataUtils的注入过程开始，首先是DataUtilsModule

```java
// 在provide方法上加上@Singleton
@Module
public class DataUtilsModule {

    @Provides
    @Singleton
    DataUtils provideDataUtils(@ApiDataUtils AbstractUtils abstractUtils) {
        return new DataUtils(abstractUtils);
    }
}
```

以及需要使用此module的Component也需要加上

```java
@Singleton
@Component(modules = DataUtilsModule.class, dependencies = AbstractUtilsComponent.class)
public interface DataUtilsComponent {
    DataUtils getDataUtils();
}
```

dagger2还有一项规定，如果一个Component被加上了`@Scope`注解，类似`@Singleton`，那么依赖这个Component的Component也需要加上`@Scope`注解，比如这里的MainActivityComponent，但是如果直接在MainActivityComponent上加上`@Singleton`会报错`error: This @Singleton component cannot depend on scoped components: @Singleton com.example.daggerdemo.di.component.DataUtilsComponent`，即单例不能依赖于单例，这是因为单例只能由自己产生，如果DataUtils在其他地方被注入了，那么MainActivityComponent将无法再进行注入，因为其依赖DataUtils是单例模式，显然这不是很符合面向对象的设计原则，因为我们可能并不知道MainActivityComponent会依赖哪些单例，所以MainActivityComponent的`@Scope`可以使用自定义的注解，自定义`@Scope`与`@Singleton`有说明区别呢，`@Singleton`相当于告诉系统，这个对象或者这个方法必定是全局单例，你看着办；而自定义`@Scope`相当于告诉系统在这个注解标注过的地方，我提供的对象是唯一的。

因此代码如下，自定义ActivityScope表明我们需要在Activity生命周期内实现单例

```java
@Scope
@Retention(RetentionPolicy.RUNTIME)
public @interface ActivityScope {}
```

MainActivityComponent注解加上`@ActivityScope`，其他地方的`@Singleton`不变

```java
@ActivityScope
@Component(dependencies = DataUtilsComponent.class)
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后运行代码可以发现MainActivity中的两个DataUtils对象是相同的，也就是在MainActivity中是单例的，但是它是不是全局单例呢，我们在创建一个SecondActivity，同样注入DataUtils

```java
public class SecondActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    DataUtils dataUtils3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        TextView textView = findViewById(R.id.text);

        DaggerSecondActivityComponent
                .builder()
                .dataUtilsComponent(DaggerDataUtilsComponent
                        .builder()
                        .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                        .build())
                .build()
                .inject(this);
        // 此时dataUtils3与dataUtils并不相同，也就说dataUtils仅在MainActivity中是单例
        Log.i(TAG, dataUtils3.toString());


        textView.setText(dataUtils3.show());
    }
}

// 同理需要一个SecondActivityComponent
@ActivityScope
@Component(dependencies = DataUtilsComponent.class)
public interface SecondActivityComponent {
    void inject(SecondActivity activity);

}
```

`@Scope`是需要成对存在的，在Module的Provide方法中使用了`@Scope`，那么对应的Component中也必须使用`@Scope`注解，当两边的`@Scope`名字一样时（比如同为`@Singleton`）, 那么该Provide方法提供的依赖将会在Component中保持“局部单例”。
而在Component中标注`@Scope`，provide方法没有标注，那么这个`@Scope`就不会起作用，而Component上的`@Scope`的作用也只是为了能顺利通过编译，就像我刚刚定义的ActivityScope一样。

`@Singleton`也是一个自定义`@Scope`，它的作用就像上面说的一样。但由于它是Dagger2中默认定义的，所以它比我们自定义Scope对了一个功能，就是编译检测，防止我们不规范的使用Scope注解，仅此而已。

如何使用Dagger2实现单例呢：

1. 依赖在Component中是单例的（供该依赖的provide方法和对应的Component类使用同一个Scope注解。）
2. 对应的Component在App中只初始化一次，每次注入依赖都使用这个Component对象。（在Application中创建该Component）

最直接的就是在自定义Application中将Component先初始化了，在通过这个Component去注入我们需要的对象，由于Component是单例的，因此通过它注入的对象也就是单例。

```java
public class MyApp extends Application {
// 注意我们之前定义的DataUtilsComponent有@Singleton注解，DataUtilsModule的provide方法有@Singleton注解
// 因此在Application中初始化的是DataUtilsComponent，这里简单使用DaggerDataUtilsComponent构造，当然也可以通过dagger注入
    private DataUtilsComponent dataUtilsComponent;

    @Override
    public void onCreate() {
        super.onCreate();
        dataUtilsComponent = DaggerDataUtilsComponent
                .builder()
                .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                .build();
    }

    public DataUtilsComponent getDataUtilsComponent() {
        return dataUtilsComponent;
    }
}
```

然后修改MainActivity和SecondActivity中inject的流程

```java
// MainActivity.java，dataUtilsComponent的参数是Application中初始化的
        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(((MyApp) getApplication()).getDataUtilsComponent())
                .build()
                .inject(this);
// SecondActivity.java
        DaggerSecondActivityComponent
                .builder()
                .dataUtilsComponent(((MyApp) getApplication()).getDataUtilsComponent())
                .build()
                .inject(this);                
```

然后我们看到在MainActivity和SecondActivity中的三个DataUtils都是相同的了。

`@Scope`是用来给开发者管理依赖的生命周期的，它可以让某个依赖在Component中保持 “局部单例”（唯一），如果将Component保存在Application中复用，则可以让该依赖在app中保持单例。 我们可以通过自定义不同的Scope注解来标记这个依赖的生命周期，所以命名是需要慎重考虑的。

* `@Singleton`告诉我们这个依赖是单例的
* `@ActivityScope`告诉我们这个依赖的生命周期和Activity相同
* `@FragmentScope`告诉我们这个依赖的生命周期和Fragment相同
* `@xxxxScope` ……

以上就是如何使用自定义`@Scope`实现单例的过程，那么如果在Application中使用注入会是什么情况呢，我们将DataUtilsComponent注入到MyApp中

```java
// 虽然这样的注入过程不是很合适，但是基本流程与在Activity中相同，首先是Module，提供对象实例
@Module
public class DataUtilsComponentModule {
    @Provides
    DataUtilsComponent provideDataUtilsComponent(){
        return DaggerDataUtilsComponent
                .builder()
                .abstractUtilsComponent(DaggerAbstractUtilsComponent.create())
                .build();
    }
}

// 然后是Component，提供注入的方法以及被注入的位置
@Component(modules = DataUtilsComponentModule.class)
public interface ApplicationComponent {
    void inject(MyApp myApp);
}

public class MyApp extends Application {

    @Inject
    DataUtilsComponent dataUtilsComponent;
// 最后是在Application的onCreate方法中注入
    @Override
    public void onCreate() {
        super.onCreate();
        DaggerApplicationComponent
                .builder()
                .build()
                .inject(this);
    }

    public DataUtilsComponent getDataUtilsComponent() {
        return dataUtilsComponent;
    }
}
```

### 1.6 @MapKey和@Lazy

`@MapKey`用于定义一些依赖集合（比如Map和Set），它的使用很简单，可以看代码注释

首先需要定义key注解

```java
// UtilsMapKey作为后续使用的注解，String代表这个注解的接受的类型为String
// unwrapValue如果为true，则此注解可接受的key类型有基本类型包装类、String、classes
// unwrapValue如果为false，则此注解可接受的key类型为其本身，这个例子可以在源码注释中找到
@MapKey(unwrapValue = true)
public @interface UtilsMapKey {
    String value();
}
```

然后是提供Map的value数据的module

```java
@Module
public class UtilsMapModule {
  // 这里首先是Provides注解，然后是IntoMap注解，最后是前面定义的MapKey注解，同时传入了Map的key值为thisiskey
    @Provides
    @IntoMap
    @UtilsMapKey("thisiskey")
    Integer provideUtilsMapValue(){
      // 返回值即为value，虽然返回值为value，但实际上注入时传入的是整个Map
        return 11;
    }
}
```

其次是MainActivityComponent加上我们定义的module

```java
@ActivityScope
@Component(dependencies = DataUtilsComponent.class, modules = UtilsMapModule.class)
public interface MainActivityComponent {
    void inject(MainActivity activity);
}
```

最后直接在MainActivity中使用即可

```java
public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    DataUtils dataUtils;

    @Inject
    DataUtils dataUtils2;

    @Inject
    Map<String, Integer> map;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TextView textView = findViewById(R.id.text);

        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(((MyApp) getApplication()).getDataUtilsComponent())
                .build()
                .inject(this);

        // 很显然，这里引入的是ApiUtils对象
        Log.i(TAG, dataUtils.toString() + dataUtils2.toString());
        Log.i(TAG, String.valueOf(dataUtils.equals(dataUtils2)));
        // 这里TextView中显示的就是 {thisiskey=11}
        textView.setText(map.toString());
        textView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, SecondActivity.class));
            }
        });
    }
}
```

`@Lazy`，这个并不是作为注解使用的，而是作为wrapper类型使用，比如下面这样

```java
    @Inject
    Lazy<DataUtils> dataUtils;
```

使用Lazy修饰的类型不会在注入的时候初始化，只能通过get方法获取实例，下面的Log日志显示了未初始化的dataUtils是什么类型

```java
public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    @Inject
    Lazy<DataUtils> dataUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        DaggerMainActivityComponent
                .builder()
                .dataUtilsComponent(((MyApp) getApplication()).getDataUtilsComponent())
                .build()
                .inject(this);
        // dataUtils是dagger.internal.DoubleCheck@3050053
        Log.i(TAG, dataUtils.toString());

        DataUtils utils = dataUtils.get();
        // utils是com.example.daggerdemo.di.model.DataUtils@26c9d90
        Log.i(TAG, utils.toString());
    }
}
```

### 1.7 @Binds和@Multibinds

`@Binds`注解与`@Provides`有异曲同工之妙，其修饰的方法都是为了提供实例，但是具体使用起来又有区别，`@Binds`只能在抽象类Module中使用，并且修饰抽象方法，为了使用`@Binds`注解，我们首先构造一个抽象类DBUtilsModule用于提供DBUtils，LocalDBUtils继承自DBUtils，我们希望不直接用到LocalDBUtils的构造方法而去生成它。

```java
// 首先这是一个Module，而且是抽象的，其次provideLocalDBUtils方法也是抽象的，用@Binds修饰，方法的参数即返回的实例
// 为了对比，加上了一个普通的Provides修饰的方法
@Module
public abstract class DBUtilsModule {
  // provideLocalDBUtils看似返回的是DBUtils，但实际返回的是LocalDBUtils的实例
    @Binds
    @LocalDBDataUtils
    abstract DBUtils provideLocalDBUtils(LocalDBUtils localDBUtils);

    @Provides
    static DBUtils provideDBUtils() {
        return new DBUtils();
    }
}
```

```java
// 因为使用provideLocalDBUtils方法，所以需要通过inject的方法提供LocalDBUtils实例，这里使用最简单的构造函数注入，
// 当然也可以使用Module提供LocalDBUtils实例，这里仅作演示
public class LocalDBUtils extends DBUtils {

    @Inject
    public LocalDBUtils() {
    }

    @Override
    public String showMessage() {
        return "This is LocalDBDataUtils";
    }
}
```

```java
// 增加一个限定符注解，为了区分provideDBUtils和provideLocalDBUtils
@Qualifier
@Retention(RetentionPolicy.RUNTIME)
public @interface LocalDBDataUtils {
}
```

```java
// 同理需要定义一个Component来提供实例方法，也可以不用这个Component，具体区别稍后再点明
@Component(modules = DBUtilsModule.class)
public interface DBUtilsComponent {

    @LocalDBDataUtils
    DBUtils getLocalDBUtils();

    DBUtils getDBUtils();
}
```

```java
// 如果加了上面的DBUtilsComponent，则ActivityComponent需要用dependencies
@Component(dependencies = DBUtilsComponent.class)
public interface SecondActivityComponent {
    void inject(SecondActivity secondActivity);
}

// 如果不加上面的DBUtilsComponent，则ActivityComponent需要modules
@Component(modules = DBUtilsModule.class)
public interface SecondActivityComponent {
    void inject(SecondActivity secondActivity);
}
```

最后再SecondActivity中使用

```java
public class SecondActivity extends AppCompatActivity {
    // 获取实例可以通过限定符注解的方法获取指定的实例
    // 比如这里@LocalDBDataUtils表明需要provideLocalDBUtils方法返回的实例
    // 如果这里不加@LocalDBDataUtils，则代表provideDBUtils返回的实例
    @LocalDBDataUtils
    @Inject
    DBUtils dbUtils;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        // 如果加了上面的DBUtilsComponent
        DaggerSecondActivityComponent
                .builder()
                .dBUtilsComponent(DaggerDBUtilsComponent.create())
                .build()
                .inject(this);        

        // 如果不加上面的DBUtilsComponent
        DaggerSecondActivityComponent
                .builder()
                .build()
                .inject(this);
        Log.i("MainActivity2", dbUtils.showMessage());
    }
}
```

通过`@Binds`修饰抽象方法会有什么区别呢，我们在加了上面的DBUtilsComponent的情况下看一下源码DaggerDBUtilsComponent

```java
public final class DaggerDBUtilsComponent implements DBUtilsComponent {
  private DaggerDBUtilsComponent() {}

  public static Builder builder() {
    return new Builder();
  }

  public static DBUtilsComponent create() {
    return new Builder().build();
  }
// Binds修饰抽象方法会导致实例在DaggerDBUtilsComponent直接构造生成
  @Override
  public DBUtils getLocalDBUtils() {
    return new LocalDBUtils();
  }
// 普通的Provides修饰是通过DBUtilsModule_ProvideDBUtilsFactory工厂类生成
  @Override
  public DBUtils getDBUtils() {
    return DBUtilsModule_ProvideDBUtilsFactory.provideDBUtils();
  }

  public static final class Builder {
    private Builder() {}

    public DBUtilsComponent build() {
      return new DaggerDBUtilsComponent();
    }
  }
}
```

简而言之，`@Binds`注解的作用还是修饰提供实例的方法，但是其修饰的方法的参数即返回的实例，我们不需要显示地调用需要地实例地构造函数，因为在生成地代码中为我们完成了这些工作，与此同时，它地效率可能会高一些，因为是直接在Component中生成的。

`@BindsInstance`比较适合与`@Component.Builder`方法一起说明，直接看代码，我们这里需要注入Application的Context，虽然显得很奇怪

首先是AppComponent，最后需要用这个调用inject方法注入到Application中

```java
// 单例模式，以及modules参数不解释
@Singleton
@Component(modules = AppModule.class)
public interface AppComponent {
// 这里出现了@Component.Builder注解，它的作用是提供自定义的方式构造此Component，根据注释要求
// 1. 必须有一个返回此Component的build方法 AppComponent build();
// 2. 可以有抽象方法作为setter方法
// 3. setter方法必须有一个参数，并且返回void、builder、builder的父类
// 4. 必须有一个setter方法用于设置dependencies（如果有的话）
// 5. 必须有setter方法用于设置modules里面有非抽象方法的非抽象module
// 6. 可以有setter方法初始化modules
// 7. 可以有@BindsInstance修饰的方法将绑定的实例传递给此Component
// 8. 可以有非抽象方法，但是如果与builder生成相关则会被忽略
// 注释说的并不是很清楚，需要自行测试其功能，此处仅演示
    @Component.Builder
    interface Builder {
        // 这里两个方法setApplication和setDBName暴露出去，在调用inject方法时初始化
        // 且这两个方法的参数application和name会被传入到AppModule的provide方法的参数中
        // 即我们实现了在外部对Component进行初始化参数设置的功能
        @BindsInstance
        Builder setApplication(Application application);

        @BindsInstance
        Builder setDBName(String name);

        AppComponent build();

    }

    void inject(MyApp app);
}
```

接下来是AppModule，提供两个方法用于提供实例，注入的对象分别是Context和Integer（Context是上下文，后续会介绍；Integer是演示，没有意义）

```java
@Module
public class AppModule {
    // 这里传入的两个参数application和name来自于AppComponent的Builder方法中的setApplication和setDBName
    @Provides
    @Singleton
    Context provideContext(Application application) {
        return application;
    }

    @Provides
    @Singleton
    Integer provideDBName(String name) {
        return name.hashCode();
    }

}
```

最后在自定义的Application中注入

```java
public class MyApp extends Application {

    @Inject
    Context context;

    @Inject
    Integer dbName;

    @Override
    public void onCreate() {
        super.onCreate();
        // 传入的参数包括this和hello.db
        // 这样就把传入的参数通过Component转换并注入到MyApp中了
        // 虽然这种方式没有意义，但是后续有用，这里仅演示BindsInstance的功能
        DaggerAppComponent
                .builder()
                .setApplication(this)
                .setDBName("hello.db")
                .build()
                .inject(this);
        Log.i("MyApp", context.toString());
        Log.i("MyApp", dbName.toString());
    }
}
```

`@MultiBinds`顾名思义，肯定与注入多个对象相关，假如我们需要在Module里提供了很多相同类型的 对象，如果我们不使用`@Qualifer`，就会导致同一类型重复绑定的错误。但是如果我们确实需要在一个Module里包含这些对象的创建，又不想创建N多的`@Qualifer`，我们就可以使用`@MultiBind`机制来达到我们的目的。

MultiBind机制允许我们为这些对象创建一个集合，这个集合必须是Set或者Map，这样在Component中，我们就可以暴露这个集合，通过集合来获取不同的对象。这个集合的创建有三种方法

1. 使用`@IntoSet`或者`@IntoMap`

```java
// 还记得上面提到的@MapKey注解吗
@Module
public class UtilsMapModule {
  // 这里首先是Provides注解，然后是IntoMap注解，最后是前面定义的MapKey注解，同时传入了Map的key值为thisiskey
    @Provides
    @IntoMap
    @UtilsMapKey("thisiskey1")
    Integer provideUtilsMapValue1(){
      // 返回值即为value，虽然返回值为value，但实际上注入时传入的是整个Map
        return 11;
    }
    // 下面加的代码没有测试过，仅演示
    @Provides
    @IntoMap
    @UtilsMapKey("thisiskey2")
    Integer provideUtilsMapValue2(){
        return 12;
    }

    @Provides
    @IntoSet
    Integer provideUtilsSetValue1(){
        return 111;
    }

    @Provides
    @IntoSet
    Integer provideUtilsSetValue2(){
        return 222;
    }    
}
```

2. 直接提供Set或者Map类型

```java
@Module
public class UtilsMapModule {
  
    @Provides
    Set<String> provideUtilsSet(){
      Set<String> utils = new HashSet<>();
      utils.add("utils1");
      utils.add("utils2");
      return utils;
    }

    @Provides
    Map<String, Integer> provideUtilsMap(){
      Map<String, Integer> utils = new HashMap<>();
      utils.put("utils-key1", 111);
      utils.put("utils-key2", 222);
      return utils;
    }

}
```

3. 使用`@MultiBinds`注解

```java
@Module
public abstract class UtilsBindModule {

    @Multibinds
    abstract Set<String> utilsSet();

    @Multibinds
    abstract Map<String, Integer> utilsMap();
}
```

MultiBinds只能用于标注抽象方法，它仅仅是告诉Component我有这么一种提供类型，让我们Component可以在Component中暴露Set或者Map类型的接口，但是不能包含具体的元素。Multibinds注解是可以和第一种集合定义混用的。

如果将UtilsBindModule单独加在某个Component的modules参数时，它并不能提供实例，而是提供一个空的实例，如果将它和另一个可以提供具体实例的Module一起加在某个Component的modules参数时，会自动获取非空实例，此时UtilsBindModule没有作用。

## 2. dagger.android进阶

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

// 11. getMapOfClassOfAndProviderOfAndroidInjectorFactoryOf方法返回的是SingletonMap，key是MainActivity.class
// value是mainActivitySubComponentFactoryProvider，简而言之还是一个map，然后需要看10中DispatchingAndroidInjector_Factory
// 的实例是如何构造的
  private Map<Class<?>, Provider<AndroidInjector.Factory<?>>>
      getMapOfClassOfAndProviderOfAndroidInjectorFactoryOf() {
    return Collections.<Class<?>, Provider<AndroidInjector.Factory<?>>>singletonMap(
        MainActivity.class, (Provider) mainActivitySubComponentFactoryProvider);
  }

// 10. getDispatchingAndroidInjectorOfActivity返回的是DispatchingAndroidInjector_Factory的实例
// 带入了参数getMapOfClassOfAndProviderOfAndroidInjectorFactoryOf()以及一个emptyMap，对的就是空map，仅包含了key和value的类型
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

// 8. DaggerAppComponent.create().inject(this)的最后一步，实际调用injectMyApplication
  @Override
  public void inject(MyApplication application) {
    injectMyApplication(application);
  }

// 9. 两个关键，MyApplication_MembersInjector类以及本地的getDispatchingAndroidInjectorOfActivity()方法
// MyApplication_MembersInjector类后续再介绍，但是本质上injectDispatchingActivityInjector方法等价于
// instance.DispatchingActivityInjector = getDispatchingAndroidInjectorOfActivity()
// 先看getDispatchingAndroidInjectorOfActivity()方法
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
      // 这里的两个方法injectEntity和injectInfo分别对应了我们在MainActivity中注入的两个对象，instance是MainActivity，
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

```java
// MainActivityModule_ProvideEntityFactory.java
// 接上文4.2
// Factory<Entity>继承自Provider，Provider之前提到过用于提供构造好的实例，通过get方法返回
public final class MainActivityModule_ProvideEntityFactory implements Factory<Entity> {
  private static final MainActivityModule_ProvideEntityFactory INSTANCE =
      new MainActivityModule_ProvideEntityFactory();

  @Override
  public Entity get() {
    return provideEntity();
  }
// 4.2-1 首先是create方法返回实例，这是饿汉式单例模式，在类初始化时，已经自行实例化
  public static MainActivityModule_ProvideEntityFactory create() {
    return INSTANCE;
  }
// 接上文7-1，provideEntity方法返回的是MainActivityModule.provideEntity()，而MainActivityModule
// 是我们定义的，MainActivityModule.provideEntity()返回new Entity()，所以我们最终得到了new出来的对象
  public static Entity provideEntity() {
    return Preconditions.checkNotNull(
        MainActivityModule.provideEntity(),
        "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

```java
// AppModule_ProvideGlobalInfoFactory.java
// 接上文4.3
public final class AppModule_ProvideGlobalInfoFactory implements Factory<String> {
  private final AppModule module;

  public AppModule_ProvideGlobalInfoFactory(AppModule module) {
    this.module = module;
  }

  @Override
  public String get() {
    return provideGlobalInfo(module);
  }
// 4.3-1 首先也是create方法，观察一下与MainActivityModule_ProvideEntityFactory的create方法的不同之处
// 这里通过构造方法返回了实例，而不是单例模式，要知道AppModule和MainActivityModule中都是加入了Singleton注解
// todo 这可能是因为需要传入参数create(AppModule module)的原因，而且AppModule是一个类而MainActivityModule
// 是一个抽象类
  public static AppModule_ProvideGlobalInfoFactory create(AppModule module) {
    return new AppModule_ProvideGlobalInfoFactory(module);
  }
// 接上文7-1，provideGlobalInfo返回的就是我们在AppModule定义的String "This is global info"
  public static String provideGlobalInfo(AppModule instance) {
    return Preconditions.checkNotNull(
        instance.provideGlobalInfo(), "Cannot return null from a non-@Nullable @Provides method");
  }
}
```

```java
// DispatchingAndroidInjector_Factory.java
// 接上文10，我们传入的参数是两个map，一个为空，另一个与MainActivity相关
public final class DispatchingAndroidInjector_Factory<T>
    implements Factory<DispatchingAndroidInjector<T>> {
  private final Provider<Map<Class<?>, Provider<AndroidInjector.Factory<?>>>>
      injectorFactoriesWithClassKeysProvider;

  private final Provider<Map<String, Provider<AndroidInjector.Factory<?>>>>
      injectorFactoriesWithStringKeysProvider;

  public DispatchingAndroidInjector_Factory(
      Provider<Map<Class<?>, Provider<AndroidInjector.Factory<?>>>>
          injectorFactoriesWithClassKeysProvider,
      Provider<Map<String, Provider<AndroidInjector.Factory<?>>>>
          injectorFactoriesWithStringKeysProvider) {
    this.injectorFactoriesWithClassKeysProvider = injectorFactoriesWithClassKeysProvider;
    this.injectorFactoriesWithStringKeysProvider = injectorFactoriesWithStringKeysProvider;
  }

  @Override
  public DispatchingAndroidInjector<T> get() {
    return new DispatchingAndroidInjector<T>(
        injectorFactoriesWithClassKeysProvider.get(),
        injectorFactoriesWithStringKeysProvider.get());
  }

  public static <T> DispatchingAndroidInjector_Factory<T> create(
      Provider<Map<Class<?>, Provider<AndroidInjector.Factory<?>>>>
          injectorFactoriesWithClassKeysProvider,
      Provider<Map<String, Provider<AndroidInjector.Factory<?>>>>
          injectorFactoriesWithStringKeysProvider) {
    return new DispatchingAndroidInjector_Factory<T>(
        injectorFactoriesWithClassKeysProvider, injectorFactoriesWithStringKeysProvider);
  }

// 10-1 newInstance返回的是DispatchingAndroidInjector，先简单说明一下这个类的作用
// DispatchingAndroidInjector类是一个完成对Activity或Fragment进行依赖注入的类，因为传入的参数包括
// injectorFactoriesWithClassKeys，这个Map根据前面的分析可知它的key就是Activity.class或者Fragment.class
// 即依赖注入的位置，它的value是一个Provider，这个Provider提供inject方法，专门用于将依赖实例注入到
// key对应的Activity或Fragment中
  public static <T> DispatchingAndroidInjector<T> newInstance(
      Map<Class<?>, Provider<AndroidInjector.Factory<?>>> injectorFactoriesWithClassKeys,
      Map<String, Provider<AndroidInjector.Factory<?>>> injectorFactoriesWithStringKeys) {
    return new DispatchingAndroidInjector<T>(
        injectorFactoriesWithClassKeys, injectorFactoriesWithStringKeys);
  }
}
```

详细分析DispatchingAndroidInjector类的功能

```java
// 注释已经说的很清楚了，DispatchingAndroidInjector类是一个完成对Activity或Fragment进行依赖注入的类
/**
 * Performs members-injection on instances of core Android types (e.g. {@link Activity}, {@link
 * Fragment}) that are constructed by the Android framework and not by Dagger. This class relies on
 * an injected mapping from each concrete class to an {@link AndroidInjector.Factory} for an {@link
 * AndroidInjector} of that class. Each concrete class must have its own entry in the map, even if
 * it extends another class which is already present in the map. Calls {@link Object#getClass()} on
 * the instance in order to find the appropriate {@link AndroidInjector.Factory}.
 *
 * @param <T> the core Android type to be injected
 */
@Beta
public final class DispatchingAndroidInjector<T> implements AndroidInjector<T> {
  private static final String NO_SUPERTYPES_BOUND_FORMAT =
      "No injector factory bound for Class<%s>";
  private static final String SUPERTYPES_BOUND_FORMAT =
      "No injector factory bound for Class<%1$s>. Injector factories were bound for supertypes "
          + "of %1$s: %2$s. Did you mean to bind an injector factory for the subtype?";

  private final Map<String, Provider<AndroidInjector.Factory<?>>> injectorFactories;

// 接上文10-1，这里就是初始化的位置，调用了merge方法
  @Inject
  DispatchingAndroidInjector(
      Map<Class<?>, Provider<AndroidInjector.Factory<?>>> injectorFactoriesWithClassKeys,
      Map<String, Provider<AndroidInjector.Factory<?>>> injectorFactoriesWithStringKeys) {
    this.injectorFactories = merge(injectorFactoriesWithClassKeys, injectorFactoriesWithStringKeys);
  }

// merge方法注释也说明了，就是将classKeyedMap的key从Class改为Class.getName的形式
  /**
   * Merges the two maps into one by transforming the values of the {@code classKeyedMap} with
   * {@link Class#getName()}.
   *
   * <p>An SPI plugin verifies the logical uniqueness of the keysets of these two maps so we're
   * assured there's no overlap.
   *
   * <p>Ideally we could achieve this with a generic {@code @Provides} method, but we'd need to have
   * <i>N</i> modules that each extend one base module.
   */
  private static <C, V> Map<String, Provider<AndroidInjector.Factory<?>>> merge(
      Map<Class<? extends C>, V> classKeyedMap, Map<String, V> stringKeyedMap) {
    if (classKeyedMap.isEmpty()) {
      @SuppressWarnings({"unchecked", "rawtypes"})
      Map<String, Provider<AndroidInjector.Factory<?>>> safeCast = (Map) stringKeyedMap;
      return safeCast;
    }

    Map<String, V> merged =
        newLinkedHashMapWithExpectedSize(classKeyedMap.size() + stringKeyedMap.size());
    merged.putAll(stringKeyedMap);
    for (Entry<Class<? extends C>, V> entry : classKeyedMap.entrySet()) {
      // put的位置，key是entry.getKey().getName()即Class.getName()，value为entry.getValue()
      // 即与classKeyedMap的value相同
      merged.put(entry.getKey().getName(), entry.getValue());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    Map<String, Provider<AndroidInjector.Factory<?>>> safeCast = (Map) merged;
    return Collections.unmodifiableMap(safeCast);
  }

// maybeInject方法即最终调用的位置，instance可以是Activity或者Fragment，如果对应我们之前的代码
// 此处应该是instance为MainActivity
  /**
   * Attempts to perform members-injection on {@code instance}, returning {@code true} if
   * successful, {@code false} otherwise.
   *
   * @throws InvalidInjectorBindingException if the injector factory bound for a class does not
   *     inject instances of that class
   */
  @CanIgnoreReturnValue
  public boolean maybeInject(T instance) {
    // injectorFactories为merge方法初始化得到的map，通过get(instance.getClass().getName())
    // 得到Provider，那么对应我们的代码，这个factoryProvider是DaggerAppComponent.java中的mainActivitySubComponentFactoryProvider，
    // 如果调用mainActivitySubComponentFactoryProvider.inject(instance)即完成依赖注入
    Provider<AndroidInjector.Factory<?>> factoryProvider =
        injectorFactories.get(instance.getClass().getName());
    if (factoryProvider == null) {
      return false;
    }
    // factory = factoryProvider.get()，这里获取到了mainActivitySubComponentFactoryProvider
    @SuppressWarnings("unchecked")
    AndroidInjector.Factory<T> factory = (AndroidInjector.Factory<T>) factoryProvider.get();
    try {
      // injector = factory.create(instance)，即返回了DaggerAppComponent.java中的MainActivitySubComponentImpl实例
      AndroidInjector<T> injector =
          checkNotNull(
              factory.create(instance), "%s.create(I) should not return null.", factory.getClass());
      // injector.inject(instance)，熟悉的味道，这里就是调用MainActivitySubComponentImpl.inject方法的最终位置
      // 在这里完成了实际的依赖注入，前提是需要调用DispatchingAndroidInjector的inject方法(此处预留伏笔)
      injector.inject(instance);
      return true;
    } catch (ClassCastException e) {
      throw new InvalidInjectorBindingException(
          String.format(
              "%s does not implement AndroidInjector.Factory<%s>",
              factory.getClass().getCanonicalName(), instance.getClass().getCanonicalName()),
          e);
    }
  }
// 倘若我们需要调用DispatchingAndroidInjector.inject方法，那么就执行了对instance的注入
// 实际调用的是maybeInject方法
  /**
   * Performs members-injection on {@code instance}.
   *
   * @throws InvalidInjectorBindingException if the injector factory bound for a class does not
   *     inject instances of that class
   * @throws IllegalArgumentException if no {@link AndroidInjector.Factory} is bound for {@code
   *     instance}
   */
  @Override
  public void inject(T instance) {
    boolean wasInjected = maybeInject(instance);
    if (!wasInjected) {
      throw new IllegalArgumentException(errorMessageSuggestions(instance));
    }
  }

  /**
   * Exception thrown if an incorrect binding is made for a {@link AndroidInjector.Factory}. If you
   * see this exception, make sure the value in your {@code @ActivityKey(YourActivity.class)} or
   * {@code @FragmentKey(YourFragment.class)} matches the type argument of the injector factory.
   */
  @Beta
  public static final class InvalidInjectorBindingException extends RuntimeException {
    InvalidInjectorBindingException(String message, ClassCastException cause) {
      super(message, cause);
    }
  }

  /** Returns an error message with the class names that are supertypes of {@code instance}. */
  private String errorMessageSuggestions(T instance) {
    List<String> suggestions = new ArrayList<>();
    for (Class<?> clazz = instance.getClass(); clazz != null; clazz = clazz.getSuperclass()) {
      if (injectorFactories.containsKey(clazz.getCanonicalName())) {
        suggestions.add(clazz.getCanonicalName());
      }
    }

    return suggestions.isEmpty()
        ? String.format(NO_SUPERTYPES_BOUND_FORMAT, instance.getClass().getCanonicalName())
        : String.format(
            SUPERTYPES_BOUND_FORMAT, instance.getClass().getCanonicalName(), suggestions);
  }
}
```

```java
// 接上文9
public final class MyApplication_MembersInjector implements MembersInjector<MyApplication> {
  private final Provider<DispatchingAndroidInjector<Activity>> dispatchingActivityInjectorProvider;

  public MyApplication_MembersInjector(
      Provider<DispatchingAndroidInjector<Activity>> dispatchingActivityInjectorProvider) {
    this.dispatchingActivityInjectorProvider = dispatchingActivityInjectorProvider;
  }

  public static MembersInjector<MyApplication> create(
      Provider<DispatchingAndroidInjector<Activity>> dispatchingActivityInjectorProvider) {
    return new MyApplication_MembersInjector(dispatchingActivityInjectorProvider);
  }

  @Override
  public void injectMembers(MyApplication instance) {
    injectDispatchingActivityInjector(instance, dispatchingActivityInjectorProvider.get());
  }
// 9-1 调用injectDispatchingActivityInjector将DispatchingAndroidInjector注入到MyApplication中
// 然后需要找到DispatchingAndroidInjector的inject方法是在哪里在什么时候被执行的，还记得最初的起点吗
// MainActivity中的AndroidInjection.inject(this)，没错，我们回来了
  public static void injectDispatchingActivityInjector(
      MyApplication instance, DispatchingAndroidInjector<Activity> dispatchingActivityInjector) {
    instance.dispatchingActivityInjector = dispatchingActivityInjector;
  }
}
```

```java
// AndroidInjection.java
  public static void inject(Activity activity) {
    checkNotNull(activity, "activity");
    Application application = activity.getApplication();
    if (!(application instanceof HasActivityInjector)) {
      throw new RuntimeException(
          String.format(
              "%s does not implement %s",
              application.getClass().getCanonicalName(),
              HasActivityInjector.class.getCanonicalName()));
    }

    AndroidInjector<Activity> activityInjector =
        ((HasActivityInjector) application).activityInjector();
    checkNotNull(activityInjector, "%s.activityInjector() returned null", application.getClass());
// 之前我们一直没有明白为什么有AndroidInjector类，为什么要调用activityInjector.inject(activity)方法
// 以及为什么需要HasActivityInjector接口,现在一切都清楚了
// activityInjector即DispatchingAndroidInjector_Factory.newInstance方法返回的DispatchingAndroidInjector实例
// 调用DispatchingAndroidInjector.inject会将DispatchingAndroidInjector_Factory.newInstance传入的Map的value中的
// 实例注入到activity中
    activityInjector.inject(activity);
  }
```

至此，与依赖注入相关的自动生成的代码已经分析完毕了，整个注入的流程也明白了，但是还遗留了几个问题：

1. 为什么要在自定义Application进行注入，以及为什么要实现接口HasActivityInjector？
2. AppComponent的module为什么必须包含AndroidInjectionModule.class？
3. MainActivitySubComponent为什么要继承AndroidInjector<MainActivity>，为什么要定义Factory继承AndroidInjector.Factory<MainActivity>？
4. MainActivityModule的subcomponents为什么是MainActivitySubComponent.class，以及为什么要定义抽象方法bindMainActivityAndroidInjectorFactory？

>  1. 为什么要在自定义Application进行注入，以及为什么要实现接口HasActivityInjector？

仔细看AndroidInjection.inject(this)的源码不难知道，activityInjector是来自于application的，为什么要依靠application，
因为当我们获取activityInjector时需要一个全局的类，其他Activity或者Fragment也能访问到，而且必须先于Activity或者Fragment被实例化，
在整个应用启动过程中只有application符合。

为什么需要实现HasActivityInjector，这是因为application目前只负责Activity的注入，需要DispatchingAndroidInjector<Activity>实例，
而activityInjector方法可以返回这个实例，`DaggerAppComponent.create().inject(this);`会将DispatchingAndroidInjector实例注入到application中。

> 2. AppComponent的module为什么必须包含AndroidInjectionModule.class？

首先看看AndroidInjectionModule的内容，抽象类加上`@Multibinds`标注的抽象方法，但是看classKeyedInjectorFactories和stringKeyedInjectorFactories两个名字就知道了，在上面的代码DispatchingAndroidInjector.java中出现过。

```java
/**
 * Contains bindings to ensure the usability of {@code dagger.android} framework classes. This
 * module should be installed in the component that is used to inject the {@link
 * android.app.Application} class.
 */
@Beta
@Module
public abstract class AndroidInjectionModule {
  @Multibinds
  abstract Map<Class<?>, AndroidInjector.Factory<?>> classKeyedInjectorFactories();

  @Multibinds
  abstract Map<String, AndroidInjector.Factory<?>> stringKeyedInjectorFactories();

  private AndroidInjectionModule() {}
}
```

MultiBinds只能用于标注抽象方法，它仅仅是告诉Component我有这么一种提供类型，让我们Component可以在Component中暴露Set或者Map类型的接口，但是不能包含具体的元素。



components所依赖的所有module里不能有重复的@Provides方法（重载，或者同返回类型的），这里还包括后面讲到的依赖的其他的Component也不能有重复的，因为Dagger无法判断你究竟想要那个作为依赖（也就是依赖迷失）



