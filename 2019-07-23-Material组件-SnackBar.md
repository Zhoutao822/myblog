---
title: Material组件-Snackbar
date: 2019-07-23 21:46:04
category:
- Android
tags:
- Android
- Material
- Snackbar
---

参考：

> [MATERIAL DESIGN](https://material.io/collections/developer-tutorials/)

Snackbar是类似与Toast的一种信息提示控件，但是与Toast不同的是Snackbar是从界面底部弹出的且支持一个点击事件，默认情况下Snackbar内部有两个子控件分别是TextView和Button，两者水平排列，TextView用于显示信息，Button用于实现点击事件。

{% asset_img snackbar0.gif %}

## 1. Snackbar使用

Snackbar属于Material组件中的一种，如果你的应用使用了Material Theme以及AppCompatActivity，则Snackbar会获得圆角、四周有margin空隙的效果。

默认情况下使用Snackbar，调用的方式也非常类似Toast

```java
// view是当前页面内的某个view，根据源码可知，make方法会找view的父view，
// 直到父view是FrameLayout或者CoordinateLayout，然后将其作为root给inflate方法调用
// inflate会加载默认的布局文件，这里根据是否使用Material Theme会加载不同的布局文件，
// 即上文我提到的效果，与此同时会将text的内容赋给布局文件中的TextView，
// 如果父view是CoordinateLayout，则Snackbar还支持右滑取消的功能，
// Snackbar.LENGTH_SHORT就类似于Toast.LENGTH_SHORT用于控制Snackbar的持续时间
Snackbar snackbar = Snackbar.make(view, text, Snackbar.LENGTH_SHORT);
snackbar.show();
```

<!-- more -->

## 2. Snackbar自定义Content

显然原生的Snackbar没有提供setContentView的方法，为了能够自定义Snackbar的布局，我们需要对Snackbar的一些参数进行修改，比如如果我们需要自定义Snackbar的margin以及自定义Snackbar的内容

```java
public class SnackbarUtil {

    private int duration;
    private View anchor;
    private ViewGroup customView;
    private int sideMargin;
    private int bottomMargin;
    private Snackbar delegete;

    /**
     * SnackbarUtil is used to create Snackbar, if you setEnableCustom(false) or in default
     * you will get the origin Snackbar from Snackbar.make(anchor, text, duration); if you
     * setEnableCustom(true) in Builder, you must add your defined customView and you can
     * modify the margin of the customView in Snackbar.
     *
     * Your customView's layout should container 2 layer of Layout, because there is some
     * UI bug if you just only use 1 layer
     *
     * <?xml version="1.0" encoding="utf-8"?>
     * <LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
     * android:layout_width="match_parent"
     * android:layout_height="match_parent">
     *
     *      <LinearLayout
     *      android:layout_width="match_parent"
     *      android:layout_height="wrap_content"
     *      android:background="@drawable/radius_background"
     *      android:orientation="vertical">
     *
     *          <TextView
     *          android:layout_width="wrap_content"
     *          android:layout_height="wrap_content"
     *          android:layout_gravity="center"
     *          android:padding="8dp"
     *          android:id="@+id/textView"
     *          android:text="hahahahh"
     *          android:textColor="@color/colorPrimaryDark"
     *          android:textSize="24sp" />
     *      </LinearLayout>
     * </LinearLayout>
     *
     * Usage:
     *
     * snackView is your customView, layout_snackbar is the layout above.
     * button is the anchor view, if your button is in CoordinatorLayout
     * the snackbar can be dismissed with swipe action.
     *
     * ViewGroup snackView = (ViewGroup) LayoutInflater.from(MainActivity.this).inflate(
     *                                      R.layout.layout_snackbar,
     *                                      new LinearLayout(MainActivity.this),
     *                                      false);
     *
     * TextView textView = snackView.findViewById(R.id.textView);
     * textView.setOnClickListener(new View.OnClickListener() {
     *
     * @param duration     Snackbar duration, default Snackbar.LENGTH_SHORT;
     * @param anchor       must need, with anchor the Snackbar will find its root;
     * @param customView   must need if setEnableCustom(true) in Builder;
     * @param sideMargin   customView left and right margin in Snackbar;
     * @param bottomMargin customView bottom margin in Snackbar;
     * @Override public void onClick(View v) {
     * Toast.makeText(MainActivity.this, "12121", Toast.LENGTH_LONG).show();
     * }
     * });
     * Snackbar snackbar = new SnackbarUtil.Builder()
     *                  .setAnchor(button)
     *                  .setBottomMargin(80)
     *                  .setDuration(Snackbar.LENGTH_SHORT)
     *                  .setText("32323")
     *                  .setCustomView(snackView)
     *                  .setSideMargin(20)
     *                  .build();
     * snackbar.show();
     */
    SnackbarUtil(int duration, View anchor, ViewGroup customView, int sideMargin, int bottomMargin) {
        this.duration = duration;
        this.anchor = anchor;
        this.customView = customView;
        this.sideMargin = sideMargin;
        this.bottomMargin = bottomMargin;
        this.delegete = Snackbar.make(anchor, "", duration);
    }

    private Snackbar create() {
        // 通过getView获取Snackbar的layout
        Snackbar.SnackbarLayout layout = (Snackbar.SnackbarLayout) delegete.getView();
        // 为了自定义margin，这里需要将Snackbar的背景设置为透明，textView可以设置为INVISIBLE也可以不设置
        // 只要没有在make加入text即可
        layout.setBackgroundColor(Color.TRANSPARENT);
        TextView textView = layout.findViewById(com.google.android.material.R.id.snackbar_text);
        textView.setVisibility(View.INVISIBLE);

        // customView为我们传入的自定义view，自定义view是如何调用inflate可以参考注释，
        // 但是customView必须包含两层layout这是因为UI上有bug，具体可以自行测试，
        // 所以我们实际设置的margin是第2层layout的margin，第2层layout有背景色，
        // 所以最终呈现出Snackbar有margin的效果，但是要知道实际上Snackbar的布局
        // 还是占据了整个底部空间
        View childLayout = customView.getChildAt(0);
        final ViewGroup.MarginLayoutParams params = (ViewGroup.MarginLayoutParams) childLayout.getLayoutParams();
        // MarginLayoutParams设置margin
        params.setMargins(params.leftMargin + sideMargin,
                params.topMargin,
                params.rightMargin + sideMargin,
                params.bottomMargin + bottomMargin);
        childLayout.setLayoutParams(params);

        // Add the view to the Snackbar's layout
        layout.addView(customView, 0);
        // Show the Snackbar
        return delegete;
    }

    // 由于参数较多，所以采用建造者模式
    public static final class Builder {
        private boolean enableCustom;
        private String text;
        private int duration;
        private View anchor;
        private ViewGroup customView;
        private int sideMargin;
        private int bottomMargin;

        Builder(boolean enableCustom) {
            this.enableCustom = enableCustom;
            this.duration = Snackbar.LENGTH_SHORT;
            this.text = "";
            this.sideMargin = 0;
            this.bottomMargin = 0;
        }

        public Builder() {
            this(false);
        }

        public Builder setEnableCustom(boolean enableCustom) {
            this.enableCustom = enableCustom;
            return this;
        }

        public Builder setText(String text) {
            this.text = text;
            return this;
        }

        public Builder setDuration(int duration) {
            this.duration = duration;
            return this;
        }

        public Builder setAnchor(View anchor) {
            this.anchor = anchor;
            return this;
        }

        public Builder setCustomView(ViewGroup customView) {
            this.customView = customView;
            return this;
        }

        public Builder setSideMargin(int sideMargin) {
            this.sideMargin = sideMargin;
            return this;
        }

        public Builder setBottomMargin(int bottomMargin) {
            this.bottomMargin = bottomMargin;
            return this;
        }

        public Snackbar build() {
            // 通过enableCustom控制是否使用自定义Content，自定义Content的点击事件需要
            // 在Snackbar的外面处理
            if (!enableCustom) {
                return Snackbar.make(anchor, text, duration);
            }
            if (anchor == null) {
                throw new IllegalArgumentException(
                        "No suitable parent found from the given view. Please provide a valid view.");
            }
            if (customView == null) {
                throw new IllegalArgumentException(
                        "No custom view found. Please provide a valid view or setEnableCustom(false).");
            }
            return new SnackbarUtil(duration, anchor, customView, sideMargin, bottomMargin).create();
        }
    }
}
```

最终效果

{% asset_img snackbar1.gif %}





