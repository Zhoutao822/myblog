---
title: Android框架-Gson
date: 2019-07-20 23:09:12
category:
- Android
tags:
- Android
- Gson
- json
---

参考：

> [Gson User Guide](https://github.com/google/gson/blob/master/UserGuide.md)


json是一种数据格式，类似与键值对的形式，常用于服务器与客户端之间数据传输，以键值对形式传输的数据在客户端进行解析时必然需要对不同的key寻找其对应的value，通常来说这种解析数据的过程非常繁琐，但是没有难度，所以Google推出了Gson这个工具，用于解析json数据并直接将其实例化。

<!-- more -->

## 1. Gson使用

以解析和风天气的数据为例，请求返回的json数据如下

```json
{
    "HeWeather6": [
        {
            "basic": {
                "cid": "CN101010100",
                "location": "北京",
                "parent_city": "北京",
                "admin_area": "北京",
                "cnty": "中国",
                "lat": "39.90498734",
                "lon": "116.4052887",
                "tz": "+8.00"
            },
            "update": {
                "loc": "2019-07-18 16:45",
                "utc": "2019-07-18 08:45"
            },
            "status": "ok",
            "now": {
                "cloud": "10",
                "cond_code": "101",
                "cond_txt": "多云",
                "fl": "35",
                "hum": "54",
                "pcpn": "0.0",
                "pres": "1000",
                "tmp": "32",
                "vis": "6",
                "wind_deg": "279",
                "wind_dir": "西风",
                "wind_sc": "1",
                "wind_spd": "3"
            }
        }
    ]
}
```

> 1.构造对应json数据的实体类，这里使用的Android Studio的插件GsonFormat，可以直接根据json数据生成代码

```java
public class WeatherEntity {

    private List<HeWeather6Bean> HeWeather6;

    public List<HeWeather6Bean> getHeWeather6() {
        return HeWeather6;
    }

    public void setHeWeather6(List<HeWeather6Bean> HeWeather6) {
        this.HeWeather6 = HeWeather6;
    }
// 重写以下toString方法，便于后续观察数据传输是否正确
    @NonNull
    @Override
    public String toString() {
        return HeWeather6.get(0).toString();
    }

    public static class HeWeather6Bean {
        /**
         * basic : {"cid":"CN101010100","location":"北京","parent_city":"北京","admin_area":"北京","cnty":"中国","lat":"39.90498734","lon":"116.4052887","tz":"+8.00"}
         * update : {"loc":"2019-07-18 16:45","utc":"2019-07-18 08:45"}
         * status : ok
         * now : {"cloud":"10","cond_code":"101","cond_txt":"多云","fl":"35","hum":"54","pcpn":"0.0","pres":"1000","tmp":"32","vis":"6","wind_deg":"279","wind_dir":"西风","wind_sc":"1","wind_spd":"3"}
         */

        private BasicBean basic;
        private UpdateBean update;
        private String status;
        private NowBean now;

        @NonNull
        @Override
        public String toString() {
            return status + " \n " + basic.toString() + " \n " + update.toString() + " \n " + now.toString();
        }

        public BasicBean getBasic() {
            return basic;
        }

        public void setBasic(BasicBean basic) {
            this.basic = basic;
        }

        public UpdateBean getUpdate() {
            return update;
        }

        public void setUpdate(UpdateBean update) {
            this.update = update;
        }

        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }

        public NowBean getNow() {
            return now;
        }

        public void setNow(NowBean now) {
            this.now = now;
        }

        public static class BasicBean {
            /**
             * cid : CN101010100
             * location : 北京
             * parent_city : 北京
             * admin_area : 北京
             * cnty : 中国
             * lat : 39.90498734
             * lon : 116.4052887
             * tz : +8.00
             */

            private String cid;
            private String location;
            private String parent_city;
            private String admin_area;
            private String cnty;
            private String lat;
            private String lon;
            private String tz;

            @NonNull
            @Override
            public String toString() {
                return "cid : " + cid + "\n" +
                        "location : " + location + "\n" +
                        "parent_city : " + parent_city + "\n" +
                        "admin_area : " + admin_area + "\n" +
                        "cnty : " + cnty + "\n" +
                        "lat : " + lat + "\n" +
                        "lon : " + lon + "\n" +
                        "tz : " + tz + "\n";
            }

            public String getCid() {
                return cid;
            }

            public void setCid(String cid) {
                this.cid = cid;
            }

            public String getLocation() {
                return location;
            }

            public void setLocation(String location) {
                this.location = location;
            }

            public String getParent_city() {
                return parent_city;
            }

            public void setParent_city(String parent_city) {
                this.parent_city = parent_city;
            }

            public String getAdmin_area() {
                return admin_area;
            }

            public void setAdmin_area(String admin_area) {
                this.admin_area = admin_area;
            }

            public String getCnty() {
                return cnty;
            }

            public void setCnty(String cnty) {
                this.cnty = cnty;
            }

            public String getLat() {
                return lat;
            }

            public void setLat(String lat) {
                this.lat = lat;
            }

            public String getLon() {
                return lon;
            }

            public void setLon(String lon) {
                this.lon = lon;
            }

            public String getTz() {
                return tz;
            }

            public void setTz(String tz) {
                this.tz = tz;
            }
        }

        public static class UpdateBean {
            /**
             * loc : 2019-07-18 16:45
             * utc : 2019-07-18 08:45
             */

            private String loc;
            private String utc;

            @NonNull
            @Override
            public String toString() {
                return "loc : " + loc + "\n" +
                        "utc : " + utc + "\n";
            }

            public String getLoc() {
                return loc;
            }

            public void setLoc(String loc) {
                this.loc = loc;
            }

            public String getUtc() {
                return utc;
            }

            public void setUtc(String utc) {
                this.utc = utc;
            }
        }

        public static class NowBean {
            /**
             * cloud : 10
             * cond_code : 101
             * cond_txt : 多云
             * fl : 35
             * hum : 54
             * pcpn : 0.0
             * pres : 1000
             * tmp : 32
             * vis : 6
             * wind_deg : 279
             * wind_dir : 西风
             * wind_sc : 1
             * wind_spd : 3
             */

            private String cloud;
            private String cond_code;
            private String cond_txt;
            private String fl;
            private String hum;
            private String pcpn;
            private String pres;
            private String tmp;
            private String vis;
            private String wind_deg;
            private String wind_dir;
            private String wind_sc;
            private String wind_spd;

            @NonNull
            @Override
            public String toString() {
                return "cloud : " + cloud + "\n" +
                        "cond_code : " + cond_code + "\n" +
                        "cond_txt : " + cond_txt + "\n" +
                        "fl : " + fl + "\n" +
                        "hum : " + hum + "\n" +
                        "pcpn : " + pcpn + "\n" +
                        "pres : " + pres + "\n" +
                        "tmp : " + tmp + "\n" +
                        "vis : " + vis + "\n" +
                        "wind_deg : " + wind_deg + "\n" +
                        "wind_dir : " + wind_dir + "\n" +
                        "wind_sc : " + wind_sc + "\n" +
                        "wind_spd : " + wind_spd + "\n";
            }

            public String getCloud() {
                return cloud;
            }

            public void setCloud(String cloud) {
                this.cloud = cloud;
            }

            public String getCond_code() {
                return cond_code;
            }

            public void setCond_code(String cond_code) {
                this.cond_code = cond_code;
            }

            public String getCond_txt() {
                return cond_txt;
            }

            public void setCond_txt(String cond_txt) {
                this.cond_txt = cond_txt;
            }

            public String getFl() {
                return fl;
            }

            public void setFl(String fl) {
                this.fl = fl;
            }

            public String getHum() {
                return hum;
            }

            public void setHum(String hum) {
                this.hum = hum;
            }

            public String getPcpn() {
                return pcpn;
            }

            public void setPcpn(String pcpn) {
                this.pcpn = pcpn;
            }

            public String getPres() {
                return pres;
            }

            public void setPres(String pres) {
                this.pres = pres;
            }

            public String getTmp() {
                return tmp;
            }

            public void setTmp(String tmp) {
                this.tmp = tmp;
            }

            public String getVis() {
                return vis;
            }

            public void setVis(String vis) {
                this.vis = vis;
            }

            public String getWind_deg() {
                return wind_deg;
            }

            public void setWind_deg(String wind_deg) {
                this.wind_deg = wind_deg;
            }

            public String getWind_dir() {
                return wind_dir;
            }

            public void setWind_dir(String wind_dir) {
                this.wind_dir = wind_dir;
            }

            public String getWind_sc() {
                return wind_sc;
            }

            public void setWind_sc(String wind_sc) {
                this.wind_sc = wind_sc;
            }

            public String getWind_spd() {
                return wind_spd;
            }

            public void setWind_spd(String wind_spd) {
                this.wind_spd = wind_spd;
            }
        }
    }
}
```

> 2.使用OkHttp构造请求

```java
OkHttpClient client = new OkHttpClient();
Request request = new Request.Builder()
        .get()
        .url(baseUrl)
        .build();
Call call = client.newCall(request);
call.enqueue(new Callback() {
    @Override
    public void onFailure(@NotNull Call call, @NotNull IOException e) {

    }

    @Override
    public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {

    }
});
```

> 3.在onResponse方法中处理请求，使用Gson对response的json数据进行实例化

```java
@Override
public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
    final String ret = response.body().string();
    runOnUiThread(new Runnable() {
        @Override
        public void run() {
            Gson gson = new Gson();
            WeatherEntity weatherEntity = gson.fromJson(ret, WeatherEntity.class);
            textView.setText(weatherEntity.toString());
        }
    });
}
```

只要对比一下就知道了，从ret到weatherEntity，完成了对json数据的实例化，我们不需要new一个对象再通过set方法赋值就可以得到一个实例，最后直接使用此实例即可。

## 2. Gson源码分析

首先new了一个Gson对象

```java
  public Gson() {
    this(Excluder.DEFAULT, FieldNamingPolicy.IDENTITY,
        Collections.<Type, InstanceCreator<?>>emptyMap(), DEFAULT_SERIALIZE_NULLS,
        DEFAULT_COMPLEX_MAP_KEYS, DEFAULT_JSON_NON_EXECUTABLE, DEFAULT_ESCAPE_HTML,
        DEFAULT_PRETTY_PRINT, DEFAULT_LENIENT, DEFAULT_SPECIALIZE_FLOAT_VALUES,
        LongSerializationPolicy.DEFAULT, null, DateFormat.DEFAULT, DateFormat.DEFAULT,
        Collections.<TypeAdapterFactory>emptyList(), Collections.<TypeAdapterFactory>emptyList(),
        Collections.<TypeAdapterFactory>emptyList());
  }
// 这里很明显，比较重要的类是TypeAdapterFactory，作用稍后再说
  Gson(final Excluder excluder, final FieldNamingStrategy fieldNamingStrategy,
      final Map<Type, InstanceCreator<?>> instanceCreators, boolean serializeNulls,
      boolean complexMapKeySerialization, boolean generateNonExecutableGson, boolean htmlSafe,
      boolean prettyPrinting, boolean lenient, boolean serializeSpecialFloatingPointValues,
      LongSerializationPolicy longSerializationPolicy, String datePattern, int dateStyle,
      int timeStyle, List<TypeAdapterFactory> builderFactories,
      List<TypeAdapterFactory> builderHierarchyFactories,
      List<TypeAdapterFactory> factoriesToBeAdded) {
    this.excluder = excluder;
    this.fieldNamingStrategy = fieldNamingStrategy;
    this.instanceCreators = instanceCreators;
    this.constructorConstructor = new ConstructorConstructor(instanceCreators);
    this.serializeNulls = serializeNulls;
    this.complexMapKeySerialization = complexMapKeySerialization;
    this.generateNonExecutableJson = generateNonExecutableGson;
    this.htmlSafe = htmlSafe;
    this.prettyPrinting = prettyPrinting;
    this.lenient = lenient;
    this.serializeSpecialFloatingPointValues = serializeSpecialFloatingPointValues;
    this.longSerializationPolicy = longSerializationPolicy;
    this.datePattern = datePattern;
    this.dateStyle = dateStyle;
    this.timeStyle = timeStyle;
    this.builderFactories = builderFactories;
    this.builderHierarchyFactories = builderHierarchyFactories;

    List<TypeAdapterFactory> factories = new ArrayList<TypeAdapterFactory>();
    // 内置的TypeAdapter，比如ObjectTypeAdapter用于处理Object类型数据
    // built-in type adapters that cannot be overridden
    factories.add(TypeAdapters.JSON_ELEMENT_FACTORY);
    factories.add(ObjectTypeAdapter.FACTORY);

    // excluder用于控制属性的是否支持序列化与反序列化，比如用@Expose修饰的属性，
    // 优先级必须在所有TypeAdapter之前
    // the excluder must precede all adapters that handle user-defined types
    factories.add(excluder);

    // 开发人员自定义的TypeAdapter，优先级相对较高
    // users' type adapters
    factories.addAll(factoriesToBeAdded);

    // 基础类型数据，包括String、Integer等包装类型
    // type adapters for basic platform types
    factories.add(TypeAdapters.STRING_FACTORY);
    factories.add(TypeAdapters.INTEGER_FACTORY);
    factories.add(TypeAdapters.BOOLEAN_FACTORY);
    factories.add(TypeAdapters.BYTE_FACTORY);
    factories.add(TypeAdapters.SHORT_FACTORY);
    TypeAdapter<Number> longAdapter = longAdapter(longSerializationPolicy);
    factories.add(TypeAdapters.newFactory(long.class, Long.class, longAdapter));
    factories.add(TypeAdapters.newFactory(double.class, Double.class,
            doubleAdapter(serializeSpecialFloatingPointValues)));
    factories.add(TypeAdapters.newFactory(float.class, Float.class,
            floatAdapter(serializeSpecialFloatingPointValues)));
    factories.add(TypeAdapters.NUMBER_FACTORY);
    factories.add(TypeAdapters.ATOMIC_INTEGER_FACTORY);
    factories.add(TypeAdapters.ATOMIC_BOOLEAN_FACTORY);
    factories.add(TypeAdapters.newFactory(AtomicLong.class, atomicLongAdapter(longAdapter)));
    factories.add(TypeAdapters.newFactory(AtomicLongArray.class, atomicLongArrayAdapter(longAdapter)));
    factories.add(TypeAdapters.ATOMIC_INTEGER_ARRAY_FACTORY);
    factories.add(TypeAdapters.CHARACTER_FACTORY);
    factories.add(TypeAdapters.STRING_BUILDER_FACTORY);
    factories.add(TypeAdapters.STRING_BUFFER_FACTORY);
    factories.add(TypeAdapters.newFactory(BigDecimal.class, TypeAdapters.BIG_DECIMAL));
    factories.add(TypeAdapters.newFactory(BigInteger.class, TypeAdapters.BIG_INTEGER));
    factories.add(TypeAdapters.URL_FACTORY);
    factories.add(TypeAdapters.URI_FACTORY);
    factories.add(TypeAdapters.UUID_FACTORY);
    factories.add(TypeAdapters.CURRENCY_FACTORY);
    factories.add(TypeAdapters.LOCALE_FACTORY);
    factories.add(TypeAdapters.INET_ADDRESS_FACTORY);
    factories.add(TypeAdapters.BIT_SET_FACTORY);
    factories.add(DateTypeAdapter.FACTORY);
    factories.add(TypeAdapters.CALENDAR_FACTORY);
    factories.add(TimeTypeAdapter.FACTORY);
    factories.add(SqlDateTypeAdapter.FACTORY);
    factories.add(TypeAdapters.TIMESTAMP_FACTORY);
    factories.add(ArrayTypeAdapter.FACTORY);
    factories.add(TypeAdapters.CLASS_FACTORY);
    
    // 集合类型优先级较低，包括Map、Collection等
    // type adapters for composite and user-defined types
    factories.add(new CollectionTypeAdapterFactory(constructorConstructor));
    factories.add(new MapTypeAdapterFactory(constructorConstructor, complexMapKeySerialization));
    this.jsonAdapterFactory = new JsonAdapterAnnotationTypeAdapterFactory(constructorConstructor);
    factories.add(jsonAdapterFactory);
    factories.add(TypeAdapters.ENUM_FACTORY);
    // 反射类型优先级最低，而这个反射类型就是我们自定义WeatherEntity的TypeAdapter
    factories.add(new ReflectiveTypeAdapterFactory(
        constructorConstructor, fieldNamingStrategy, excluder, jsonAdapterFactory));

    this.factories = Collections.unmodifiableList(factories);
  }
```

然后直接看fromJson方法，传入的参数为String和.class，返回值为.class的实例

```java
  public <T> T fromJson(String json, Class<T> classOfT) throws JsonSyntaxException {
    // 此处只要分析fromJson方法
    Object object = fromJson(json, (Type) classOfT);
    // wrap仅仅把基础类型转为包装类型，cast用于类型转换，把Object类型转为object的实际类型
    return Primitives.wrap(classOfT).cast(object);
  }

  public <T> T fromJson(String json, Type typeOfT) throws JsonSyntaxException {
    if (json == null) {
      return null;
    }
    // 通过StringReader将String类型的json数据转为StringReader
    StringReader reader = new StringReader(json);
    // 又调用fromJson方法
    T target = (T) fromJson(reader, typeOfT);
    return target;
  }

  public <T> T fromJson(Reader json, Type typeOfT) throws JsonIOException, JsonSyntaxException {
    // 又将StringReader转为JsonReader
    JsonReader jsonReader = newJsonReader(json);
    // 继续调用fromJson
    T object = (T) fromJson(jsonReader, typeOfT);
    assertFullConsumption(object, jsonReader);
    return object;
  }

  public <T> T fromJson(JsonReader reader, Type typeOfT) throws JsonIOException, JsonSyntaxException {
    boolean isEmpty = true;
    boolean oldLenient = reader.isLenient();
    reader.setLenient(true);
    try {
      // 核心代码在try catch内，我们得到的JsonReader需要通过TypeAdapter的read方法转为Java对象
      // 所以接下来需要分析JsonReader的功能，以及这里默认使用的TypeAdapter的功能
      reader.peek();
      isEmpty = false;
      TypeToken<T> typeToken = (TypeToken<T>) TypeToken.get(typeOfT);
      TypeAdapter<T> typeAdapter = getAdapter(typeToken);
      T object = typeAdapter.read(reader);
      return object;
    } catch (EOFException e) {
      /*
       * For compatibility with JSON 1.5 and earlier, we return null for empty
       * documents instead of throwing.
       */
      if (isEmpty) {
        return null;
      }
      throw new JsonSyntaxException(e);
    } catch (IllegalStateException e) {
      throw new JsonSyntaxException(e);
    } catch (IOException e) {
      // TODO(inder): Figure out whether it is indeed right to rethrow this as JsonSyntaxException
      throw new JsonSyntaxException(e);
    } catch (AssertionError e) {
      throw new AssertionError("AssertionError (GSON " + GsonBuildConfig.VERSION + "): " + e.getMessage(), e);
    } finally {
      reader.setLenient(oldLenient);
    }
  }
```

JsonReader并不是直接通过String解析出来的，首先经过了StringReader，那么先看看StringReader的构造，StringReader继承自Reader，需要实现read方法，read方法一般是用于读取字符到buffer中

```java
// StringReader.java 这里只保存了String的值和长度
    /**
     * Creates a new string reader.
     *
     * @param s  String providing the character stream.
     */
    public StringReader(String s) {
        this.str = s;
        this.length = s.length();
    }
```

JsonReader并不是继承自Reader，JsonReader需要配合TypeAdapter使用

```java
// Gson.java newJsonReader将StringReader转为JsonReader对象，DEFAULT_LENIENT为false，暂时不明白
  /**
   * Returns a new JSON reader configured for the settings on this Gson instance.
   */
  public JsonReader newJsonReader(Reader reader) {
    JsonReader jsonReader = new JsonReader(reader);
    jsonReader.setLenient(lenient);
    return jsonReader;
  }
```

getAdapter方法如何获取到TypeAdapter

```java
  /**
   * Returns the type adapter for {@code} type.
   *
   * @throws IllegalArgumentException if this GSON cannot serialize and
   *     deserialize {@code type}.
   */
  @SuppressWarnings("unchecked")
  public <T> TypeAdapter<T> getAdapter(TypeToken<T> type) {
    // 初始情况下typeTokenCache为空
    TypeAdapter<?> cached = typeTokenCache.get(type == null ? NULL_KEY_SURROGATE : type);
    if (cached != null) {
      return (TypeAdapter<T>) cached;
    }
    // calls初始也为空
    Map<TypeToken<?>, FutureTypeAdapter<?>> threadCalls = calls.get();
    boolean requiresThreadLocalCleanup = false;
    if (threadCalls == null) {
      threadCalls = new HashMap<TypeToken<?>, FutureTypeAdapter<?>>();
      calls.set(threadCalls);
      requiresThreadLocalCleanup = true;
    }
    // ThreadLocal在这里是防止老是执行for (TypeAdapterFactory factory : factories) 递归查找，
    // 如果不用ThreadLocal干预的话，就会导致堆栈溢出
    // the key and value type parameters always agree
    FutureTypeAdapter<T> ongoingCall = (FutureTypeAdapter<T>) threadCalls.get(type);
    if (ongoingCall != null) {
      return ongoingCall;
    }

    try {
      FutureTypeAdapter<T> call = new FutureTypeAdapter<T>();
      threadCalls.put(type, call);
      // TypeAdapter是从Gson初始化的factories中按照顺序遍历得到的，
      // 所以接下来需要看这里使用的是哪个TypeAdapterFactory
      for (TypeAdapterFactory factory : factories) {
        TypeAdapter<T> candidate = factory.create(this, type);
        if (candidate != null) {
          call.setDelegate(candidate);
          typeTokenCache.put(type, candidate);
          return candidate;
        }
      }
      throw new IllegalArgumentException("GSON (" + GsonBuildConfig.VERSION + ") cannot handle " + type);
    } finally {
      threadCalls.remove(type);

      if (requiresThreadLocalCleanup) {
        calls.remove();
      }
    }
  }
```

ReflectiveTypeAdapterFactory的create方法得到我们处理WeatherEntity的TypeAdapter

```java
// ReflectiveTypeAdapterFactory.java create方法返回的是Adapter
  @Override public <T> TypeAdapter<T> create(Gson gson, final TypeToken<T> type) {
    Class<? super T> raw = type.getRawType();

    if (!Object.class.isAssignableFrom(raw)) {
      return null; // it's a primitive!
    }

    ObjectConstructor<T> constructor = constructorConstructor.get(type);
    return new Adapter<T>(constructor, getBoundFields(gson, type, raw));
  }

  // Adapter的read方法就是返回WeatherEntity的位置
    Adapter(ObjectConstructor<T> constructor, Map<String, BoundField> boundFields) {
      this.constructor = constructor;
      this.boundFields = boundFields;
    }
    // read实际上是被递归调用的
    @Override public T read(JsonReader in) throws IOException {
      // JsonReader中保存了我们需要解析的字符串数据，所以也封装了一些读取的函数，
      // 通过peek判断JsonReader是否已经读到结尾了来结束解析的过程
      if (in.peek() == JsonToken.NULL) {
        in.nextNull();
        return null;
      }
      // 两个类ObjectConstructor和BoundField，看名字就知道了ObjectConstructor
      // 用于构造实例，BoundField用于指定属性
      T instance = constructor.construct();
    
      try {
        // JsonReader有几个方法，比如beginObject和beginArray，表明要开始解析
        // JsonReader的数据了，beginObject表明需要解析为对象，beginArray表明需要解析为
        // 数组
        in.beginObject();
        // hasNext表明数据是否到达结尾
        while (in.hasNext()) {
          // nextName可以获取json数据中的key
          String name = in.nextName();
          // 通过将name转为BoundField，为后续生成属性做铺垫
          BoundField field = boundFields.get(name);
          // 如果不能生成此属性或者不允许反序列化，则跳过此key对应的value数据
          if (field == null || !field.deserialized) {
            in.skipValue();
          } else {
            // 然后需要对属性进行赋值，因此需要看BoundField的read方法做了些什么
            field.read(in, instance);
          }
        }
      } catch (IllegalStateException e) {
        throw new JsonSyntaxException(e);
      } catch (IllegalAccessException e) {
        throw new AssertionError(e);
      }
      in.endObject();
      // 最后返回我们的实例
      return instance;
    }

// BoundField来源于getBoundFields方法
  private Map<String, BoundField> getBoundFields(Gson context, TypeToken<?> type, Class<?> raw) {
    Map<String, BoundField> result = new LinkedHashMap<String, BoundField>();
    if (raw.isInterface()) {
      return result;
    }

    Type declaredType = type.getType();
    while (raw != Object.class) {
      // 首先获得我们自定义WeatherEntity的属性
      Field[] fields = raw.getDeclaredFields();
      for (Field field : fields) {
        // 对于每一个属性我们通过Excluder判断是否有@Expose或者其他注解修饰
        // 根据注解的要求保存这个属性是否支持序列化serialize和反序列化deserialize
        boolean serialize = excludeField(field, true);
        boolean deserialize = excludeField(field, false);
        if (!serialize && !deserialize) {
          continue;
        }
        accessor.makeAccessible(field);
        Type fieldType = $Gson$Types.resolve(type.getType(), raw, field.getGenericType());
        // 因为Gson支持序列化时指定key的名称，所以会有一些替代名称，替代名称可以有多个，因此需要
        // 通过getFieldNames获取属性的所有序列化时的名称列表（第一个为属性名，其他可以是设置的替代名称）
        List<String> fieldNames = getFieldNames(field);
        BoundField previous = null;
        for (int i = 0, size = fieldNames.size(); i < size; ++i) {
          String name = fieldNames.get(i);
          if (i != 0) serialize = false; // only serialize the default name
          // result的value boundField是通过createBoundField得到的，且只有第一个名称允许序列化
          BoundField boundField = createBoundField(context, field, name,
              TypeToken.get(fieldType), serialize, deserialize);
          BoundField replaced = result.put(name, boundField);
          if (previous == null) previous = replaced;
        }
        if (previous != null) {
          throw new IllegalArgumentException(declaredType
              + " declares multiple JSON fields named " + previous.name);
        }
      }
      type = TypeToken.get($Gson$Types.resolve(type.getType(), raw, raw.getGenericSuperclass()));
      raw = type.getRawType();
    }
    return result;
  }

// createBoundField返回我们需要的BoundField
  private ReflectiveTypeAdapterFactory.BoundField createBoundField(
      final Gson context, final Field field, final String name,
      final TypeToken<?> fieldType, boolean serialize, boolean deserialize) {
    final boolean isPrimitive = Primitives.isPrimitive(fieldType.getRawType());
    // special casing primitives here saves ~5% on Android...
    JsonAdapter annotation = field.getAnnotation(JsonAdapter.class);
    TypeAdapter<?> mapped = null;
    if (annotation != null) {
      mapped = jsonAdapterFactory.getTypeAdapter(
          constructorConstructor, context, fieldType, annotation);
    }
    // mapped也是通过getAdapter获取的，但是fieldType已经改变了，变成了我们定义的实体类的下一级
    final boolean jsonAdapterPresent = mapped != null;
    if (mapped == null) mapped = context.getAdapter(fieldType);

    final TypeAdapter<?> typeAdapter = mapped;
    return new ReflectiveTypeAdapterFactory.BoundField(name, serialize, deserialize) {
      @SuppressWarnings({"unchecked", "rawtypes"}) // the type adapter and field type always agree
      @Override void write(JsonWriter writer, Object value)
          throws IOException, IllegalAccessException {
        Object fieldValue = field.get(value);
        TypeAdapter t = jsonAdapterPresent ? typeAdapter
            : new TypeAdapterRuntimeTypeWrapper(context, typeAdapter, fieldType.getType());
        t.write(writer, fieldValue);
      }
      // 在调用read方法时就产生了递归
      @Override void read(JsonReader reader, Object value)
          throws IOException, IllegalAccessException {
        // 由于属性的类型不同，此处的typeAdapter变为从factories遍历获取，
        // 如果我们这里的reader是String，那么typeAdapter为TypeAdapters.STRING_FACTORY
        // 然后按照TypeAdapters.STRING_FACTORY的逻辑读取数据
        Object fieldValue = typeAdapter.read(reader);
        if (fieldValue != null || !isPrimitive) {
          // set方法把值fieldValue赋给对象value
          field.set(value, fieldValue);
        }
      }
      @Override public boolean writeField(Object value) throws IOException, IllegalAccessException {
        if (!serialized) return false;
        Object fieldValue = field.get(value);
        return fieldValue != value; // avoid recursion for example for Throwable.cause
      }
    };
  }
```

我们目前知道了属性是通过getDeclaredFields拿到的，然后通过递归的方式调用typeAdapter的read方法，然后将从JsonReader中获取到的值赋给属性，关键是属性实例是如何得到的`T instance = constructor.construct();`，默认情况下是ConstructorConstructor，通过ConstructorConstructor构造某个属性的实例

```java
// ConstructorConstructor.java 默认instanceCreators为空
  public <T> ObjectConstructor<T> get(TypeToken<T> typeToken) {
    final Type type = typeToken.getType();
    final Class<? super T> rawType = typeToken.getRawType();

    // first try an instance creator

    @SuppressWarnings("unchecked") // types must agree
    final InstanceCreator<T> typeCreator = (InstanceCreator<T>) instanceCreators.get(type);
    if (typeCreator != null) {
      return new ObjectConstructor<T>() {
        @Override public T construct() {
          return typeCreator.createInstance(type);
        }
      };
    }

    // Next try raw type match for instance creators
    @SuppressWarnings("unchecked") // types must agree
    final InstanceCreator<T> rawTypeCreator =
        (InstanceCreator<T>) instanceCreators.get(rawType);
    if (rawTypeCreator != null) {
      return new ObjectConstructor<T>() {
        @Override public T construct() {
          return rawTypeCreator.createInstance(type);
        }
      };
    }

    // 我们构造的实例一般是通过newDefaultConstructor得到的
    ObjectConstructor<T> defaultConstructor = newDefaultConstructor(rawType);
    if (defaultConstructor != null) {
      return defaultConstructor;
    }

    // newDefaultImplementationConstructor用于构造Map和List以及它们的父类接口的实例
    ObjectConstructor<T> defaultImplementation = newDefaultImplementationConstructor(type, rawType);
    if (defaultImplementation != null) {
      return defaultImplementation;
    }

    // finally try unsafe
    return newUnsafeAllocator(type, rawType);
  }

  private <T> ObjectConstructor<T> newDefaultConstructor(Class<? super T> rawType) {
    try {
      // getDeclaredConstructor通过反射得到目标类的构造函数
      final Constructor<? super T> constructor = rawType.getDeclaredConstructor();
      if (!constructor.isAccessible()) {
        accessor.makeAccessible(constructor);
      }
      return new ObjectConstructor<T>() {
        @SuppressWarnings("unchecked") // T is the same raw type as is requested
        @Override public T construct() {
          try {
            Object[] args = null;
            // 返回初始参数都为null的实例
            return (T) constructor.newInstance(args);
          } catch (InstantiationException e) {
            // TODO: JsonParseException ?
            throw new RuntimeException("Failed to invoke " + constructor + " with no args", e);
          } catch (InvocationTargetException e) {
            // TODO: don't wrap if cause is unchecked!
            // TODO: JsonParseException ?
            throw new RuntimeException("Failed to invoke " + constructor + " with no args",
                e.getTargetException());
          } catch (IllegalAccessException e) {
            throw new AssertionError(e);
          }
        }
      };
    } catch (NoSuchMethodException e) {
      return null;
    }
  }
```

读取json字符串的工作是由JsonReader完成的







