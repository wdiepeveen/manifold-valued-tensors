import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (1.0, 0.5, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,0.995);
pen SpherePen = rgb(0.85,0.85,0.85)+opacity(0.6);
pen SphereLinePen = rgb(0.75,0.75,0.75)+opacity(0.6)+linewidth(0.5pt);
draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);

/*
  Colors
*/
pen pointStyle1 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(3.5pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (0.8608982822013722,-0.4725513321179279,-0.18854544867564008), pointStyle1);
dot( (0.7648000933550485,-0.4603926813335605,0.45068769250846336), pointStyle1);
dot( (0.28349486299755655,0.16552054652461962,0.9445811830288531), pointStyle1);
dot( (0.8120710958917794,0.3048940575972932,0.4975742646670772), pointStyle1);
dot( (0.18778687563278418,0.22362224200939335,0.9564147542874741), pointStyle1);
dot( (0.5094874979550965,0.06645057888276265,0.8579083925417683), pointStyle1);
dot( (0.6119867454332003,0.2650819275284312,0.745119987057045), pointStyle1);
