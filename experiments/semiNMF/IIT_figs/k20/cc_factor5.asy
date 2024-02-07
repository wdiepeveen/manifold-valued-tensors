import settings;
import three;
surface ellipsoid(triple v1,triple v2,triple v3,real l1,real l2, real l3, triple pos=O) {
  transform3 T = identity(4);
  T[0][0] = l1*v1.x;
  T[1][0] = l1*v1.y;
  T[2][0] = l1*v1.z;
  T[0][1] = l2*v2.x;
  T[1][1] = l2*v2.y;
  T[2][1] = l2*v2.z;
  T[0][2] = l3*v3.x;
  T[1][2] = l3*v3.y;
  T[2][2] = l3*v3.z;
  T[0][3] = pos.x;
  T[1][3] = pos.y;
  T[2][3] = pos.z;
  return T*unitsphere;
}

size(200);

real gDx=1.5;
real gDy=1.5;
real gDz=1.5;

currentprojection=perspective(up=Y, camera = (gDx*-2.0,gDy*6.0,gDz*14.0), target = (gDx*1.5,gDy*1.5,gDz*0.0) );
currentlight=Viewport;

  draw(  ellipsoid( (-0.9548665512318815,-0.1876178518180529,0.23028115645382669), (0.028757873289435035,0.713231104957005,0.7003387577777475), (0.29563973694752493,-0.675352450651501,0.6756450350121802), 0.822308789343159, 0.8765583704451156, 0.9563884598034005,  (gDx*0, gDy*0, gDz*0)), rgb(0.28270251775544086,0.13940084335733005,0.4562709171656971)  );
  draw(  ellipsoid( (-0.9052774746456658,-0.19602168932093084,0.37689281130705365), (0.08932640922530884,0.7795143531499736,0.6199823915629304), (-0.4153233517562885,0.5949225752557609,-0.6881668714323407), 0.8229075718625986, 0.8952943339595063, 0.9814191246369288,  (gDx*0, gDy*0, gDz*1)), rgb(0.28127464318770423,0.15708056357041206,0.4701347233306428)  );
  draw(  ellipsoid( (-0.8250165048157794,-0.1382290778124914,0.5479420487867888), (0.12140069531664457,0.9036308175419654,0.41074714456107225), (-0.551914520517616,0.40539371928478013,-0.7287292325728829), 0.7936646272653135, 0.8716208590378242, 0.9530197507873971,  (gDx*0, gDy*0, gDz*2)), rgb(0.2807359425356241,0.1618313374219545,0.4736743259613194)  );
  draw(  ellipsoid( (-0.6955899272741403,-0.09805635939292827,0.7117159570062772), (-0.04701192599927541,0.9947312778163433,0.0911019415144597), (0.7168992480736706,-0.02991045496929395,0.6965348755051218), 0.7170872958185075, 0.7986562537484925, 0.8945021899272942,  (gDx*0, gDy*0, gDz*3)), rgb(0.27668656992819085,0.18753464481700746,0.49137499522640826)  );
  draw(  ellipsoid( (-0.4770553178282767,0.5641405061748864,0.673916695908704), (-0.6736698811034978,0.25773612861578304,-0.6926333657138876), (-0.5644352177404968,-0.7844218108255866,0.25709007696787073), 0.6862828832636411, 0.7471328009782192, 0.8295523500800074,  (gDx*0, gDy*1, gDz*0)), rgb(0.2801236618700905,0.16663917454528004,0.4771728195544098)  );
  draw(  ellipsoid( (0.5638300597113405,-0.4004369756886713,-0.7223197991660042), (0.3426940119261952,-0.6823154260597492,0.6457603840054232), (-0.7514362767100309,-0.6116337857429182,-0.24748259369489453), 0.6814957165094753, 0.769845565800874, 0.8260702806869793,  (gDx*0, gDy*1, gDz*1)), rgb(0.2796736507891274,0.16988110459403927,0.47948499102620123)  );
  draw(  ellipsoid( (-0.5565557603294449,0.22689171372125963,0.799228275205991), (0.2340777950103294,-0.8801917394491896,0.4128802340734081), (-0.7971532296410874,-0.4168724649698513,-0.43676547073085353), 0.6255710745180417, 0.7004901073949352, 0.8266917082326672,  (gDx*0, gDy*1, gDz*2)), rgb(0.2676970317156612,0.22446908178968913,0.512465014952188)  );
  draw(  ellipsoid( (-0.4646380653729919,-0.051064678399534214,0.8840270735822492), (0.3652503190897232,-0.9205033618513554,0.1388011715556031), (-0.8066620560131104,-0.38738347853297095,-0.44635229129959303), 0.5240962803410261, 0.5768273630429471, 0.7997372144357275,  (gDx*0, gDy*1, gDz*3)), rgb(0.23583499661033455,0.30882913559284764,0.5427789328812678)  );
  draw(  ellipsoid( (-0.1733174908144252,0.36664133294113654,0.9140761348798789), (0.6228856996613965,-0.678111822437604,0.3900996813991059), (0.7628725008381967,0.6369760507660379,-0.11084700363734415), 0.4194006792569562, 0.47107388429364516, 0.7270970367072744,  (gDx*0, gDy*2, gDz*0)), rgb(0.2101885703890398,0.3644001804170824,0.552284440152947)  );
  draw(  ellipsoid( (-0.29696110892196387,0.25098086833076194,0.9213157458329766), (-0.5282028118294748,0.7606115905045288,-0.3774543654080686), (-0.7954972591802403,-0.5987308344507644,-0.09330326100741186), 0.41444852297105644, 0.4628545247103399, 0.7248497239571454,  (gDx*0, gDy*2, gDz*1)), rgb(0.2081441574378743,0.36877944686870684,0.5527883628347401)  );
  draw(  ellipsoid( (-0.2709390009620161,-0.02811435851528416,0.9621858659339051), (0.5191210267055986,-0.8460268615491295,0.12145743768685073), (0.8106203904355174,0.5323984714315056,0.24381642731277448), 0.3704674062304064, 0.41783697597676966, 0.7162228897431044,  (gDx*0, gDy*2, gDz*2)), rgb(0.19132162173238498,0.4055157038641537,0.555987767498092)  );
  draw(  ellipsoid( (-0.14943363462031767,-0.36355802557702177,0.9195081026737605), (0.5540831661537289,-0.8010102475511631,-0.22665927799912286), (0.8189392125360837,0.4756134411016593,0.32113925455200537), 0.2899750054557622, 0.34641807642934497, 0.6916360374232287,  (gDx*0, gDy*2, gDz*3)), rgb(0.16601429199143422,0.46520070683454784,0.5581278397174589)  );
  draw(  ellipsoid( (0.016717794934600214,0.0873545746448918,0.9960369941026965), (0.5900174580756534,-0.8051050443589387,0.060706397634317996), (0.8072173898637648,0.5866643383029189,-0.06500030511566393), 0.20903981694774815, 0.2770342068166208, 0.6128723137584364,  (gDx*0, gDy*3, gDz*0)), rgb(0.1483965692300861,0.5097028366583959,0.5571598264072957)  );
  draw(  ellipsoid( (-0.03391767427481032,-0.06751014186352484,0.9971419016957187), (0.5731918438520092,-0.8186332099961018,-0.03592739390788449), (0.8187189392662916,0.5703350315709577,0.06646239726065471), 0.19758718818267776, 0.2606331056723663, 0.6101911647021248,  (gDx*0, gDy*3, gDz*1)), rgb(0.14372068651539888,0.5217914418243871,0.5563688835909361)  );
  draw(  ellipsoid( (-0.036608570473617146,-0.2665760190309332,0.9631183928497566), (0.5658658206240044,-0.7998996985004027,-0.199890833678003), (0.8236842147433691,0.5376780620562682,0.18012944225199634), 0.17256492402481574, 0.2376821050027086, 0.6049642954183865,  (gDx*0, gDy*3, gDz*2)), rgb(0.13465894814840038,0.5459858533012772,0.5538788554016776)  );
  draw(  ellipsoid( (-0.028742713343261107,-0.4286612380779195,0.9030079730540436), (0.5641493614991158,-0.7527098492593578,-0.3393573054290933), (-0.8251723179735291,-0.49967732167615975,-0.2634638871889897), 0.13700309249368073, 0.21090243382070886, 0.5958301156334153,  (gDx*0, gDy*3, gDz*3)), rgb(0.12371848692121387,0.5806768398018529,0.5476758154374056)  );
  draw(  ellipsoid( (0.8422488646590544,0.5217346258141453,-0.13568282944834137), (-0.3459666029194517,0.7161440470547265,0.6061722639089091), (0.413429509886399,-0.46360617348634375,0.783674266682556), 0.803690537840117, 0.9034576178375688, 0.9735781014227011,  (gDx*1, gDy*0, gDz*0)), rgb(0.2798024009213985,0.16895357427256794,0.47882346868726355)  );
  draw(  ellipsoid( (0.853036790238381,0.5064043186979018,-0.1260273799771872), (-0.31081501115792426,0.6870240934227635,0.656804326946412), (0.4191923941804039,-0.5211070533592339,0.7434548914362976), 0.8105816340290586, 0.910840867807707, 0.9999999999999997,  (gDx*1, gDy*0, gDz*1)), rgb(0.2779867944431983,0.18050654921596027,0.4867888940630094)  );
  draw(  ellipsoid( (0.8632592292639036,0.4648059604308244,-0.19682205729711816), (-0.221760507425511,0.6995253372991006,0.6793280355049095), (-0.45343773600911363,0.5427888370485877,-0.7069473091669358), 0.8017797125788222, 0.8832808880026136, 0.9863728341140356,  (gDx*1, gDy*0, gDz*2)), rgb(0.27833206825606743,0.178449342893316,0.4853994259889159)  );
  draw(  ellipsoid( (-0.8400716174555607,-0.37203174528035005,0.39480635512774476), (-0.08288485934791327,0.8072673816046658,0.5843367819057976), (0.5361061253487746,-0.4581612762999549,0.7089982138642669), 0.767646876752303, 0.8233906166576652, 0.9292110702151608,  (gDx*1, gDy*0, gDz*3)), rgb(0.2798085790793642,0.1689090661330972,0.47879172511204793)  );
  draw(  ellipsoid( (0.8688333368248204,0.46087750740358097,-0.18088824171656448), (-0.3240770413408881,0.8056006821807739,0.49596533361275175), (-0.37430295763937416,0.3722894895633396,-0.8492925478673912), 0.831090800872401, 0.8609201491540148, 0.9397197144025243,  (gDx*1, gDy*1, gDz*0)), rgb(0.2832269832924779,0.12045577630499474,0.44030235419011604)  );
  draw(  ellipsoid( (0.8619622202534541,0.3669457778286065,-0.3498169907101353), (-0.17976746407562685,0.8664001205680049,0.46586960615559037), (0.4740303679254675,-0.33867628675994665,0.8127690834861646), 0.8208155687287071, 0.8515420797965716, 0.948025357541169,  (gDx*1, gDy*1, gDz*1)), rgb(0.2827759299385668,0.1379927920595199,0.4551205116906585)  );
  draw(  ellipsoid( (0.7903661599959815,0.1088173977446023,-0.6028931141432117), (0.0467127766425938,0.9705295712687402,0.23641122602629128), (0.6108512500040787,-0.2150142442746712,0.7619905676105488), 0.7755081515516188, 0.8053300869599269, 0.9057422696497417,  (gDx*1, gDy*1, gDz*2)), rgb(0.28221047915032343,0.14689249404973376,0.46227856999873473)  );
  draw(  ellipsoid( (0.6495453684060102,0.025244268473023124,-0.759903639477771), (0.06534326819846434,-0.9976043207829379,0.022712913867596784), (0.7575097832262258,0.0644076553250874,0.649638809071688), 0.6811256689588782, 0.7194957874127565, 0.837612193236648,  (gDx*1, gDy*1, gDz*3)), rgb(0.2775252979328302,0.18306159994135787,0.48847141152279094)  );
  draw(  ellipsoid( (-0.4642104610657982,0.7033095286602302,0.5383905225139128), (0.2873270584000051,-0.45540581744444153,0.8426438767053137), (-0.8378256437638325,-0.5458582676240084,-0.009324286473293102), 0.6588817507996351, 0.7111918741788178, 0.773879431858515,  (gDx*1, gDy*2, gDz*0)), rgb(0.2822680977831825,0.1461820548768388,0.4617216851972805)  );
  draw(  ellipsoid( (0.5219853373160963,-0.5505913541482158,-0.6514449081574244), (0.06679922073193342,-0.7350173038711167,0.6747498996810898), (0.8503347409912215,0.3957255662037371,0.3468891818954923), 0.6410854254525684, 0.6895084083908454, 0.7835108863527918,  (gDx*1, gDy*2, gDz*1)), rgb(0.2788055371081523,0.17561260138027185,0.4834799576696529)  );
  draw(  ellipsoid( (0.5181424893418822,-0.2815551705058355,-0.8076230845512201), (0.1779700163693483,-0.8881032305915183,0.42379160572904534), (0.8365733883002089,0.36331713113248504,0.41005564038745873), 0.5816347295155274, 0.611144131425974, 0.7713167065245088,  (gDx*1, gDy*2, gDz*2)), rgb(0.26395449093412104,0.23671148575100134,0.5183365648512119)  );
  draw(  ellipsoid( (-0.2691648531781236,-0.3720416585658497,0.8883328689771481), (0.49062057829612643,-0.8466872560390918,-0.20594207586932817), (0.8287591507585143,0.3804020172824601,0.4104297446353776), 0.47464698626493884, 0.5041322181163631, 0.7320427349584369,  (gDx*1, gDy*2, gDz*3)), rgb(0.23061970007894983,0.32042437011062963,0.545309116036773)  );
  draw(  ellipsoid( (-0.12021382308197404,0.27370685596455746,0.9542710274015516), (0.5457614107847488,-0.7847339567361826,0.2938317539739634), (-0.8292726446942865,-0.5561269406737748,0.05504277081117395), 0.38998092840669574, 0.42209094886031784, 0.6572239540016729,  (gDx*1, gDy*3, gDz*0)), rgb(0.21321145063385039,0.35793940810511404,0.5514843284112426)  );
  draw(  ellipsoid( (-0.14713828480908503,0.0032142149752750503,0.9891107086497108), (0.5195082189550156,-0.850707894090307,0.08004554560138046), (0.8417015715689432,0.5256289068863387,0.12350188931285264), 0.3760313841986887, 0.4051199755423071, 0.6532879779565697,  (gDx*1, gDy*3, gDz*1)), rgb(0.20692601232374647,0.37139320655118996,0.5530767506978597)  );
  draw(  ellipsoid( (-0.07753717859991925,-0.3297304220269947,0.940885665064924), (0.5400414298318499,-0.8071775574796057,-0.2383687159977774), (0.8380592103186598,0.489634802164999,0.24065435900665996), 0.3256287399432308, 0.36736685291702076, 0.644066566574955,  (gDx*1, gDy*3, gDz*2)), rgb(0.1878114746023558,0.4134299831119168,0.5564711815087939)  );
  draw(  ellipsoid( (-0.033464636590447495,-0.512091051218683,0.8582790183615195), (0.5549947309754295,-0.7237061315679093,-0.4101588518129562), (0.8311804657758626,0.46261451597713166,0.30842639789723647), 0.25104197265420175, 0.31553795777095883, 0.6245915056922282,  (gDx*1, gDy*3, gDz*3)), rgb(0.16308387684443304,0.47248489567860796,0.5581450809270098)  );
  draw(  ellipsoid( (0.7874151417783505,0.5976308980177321,-0.15104537143752336), (0.4720258484244432,-0.7421727861959247,-0.47578477681548687), (0.3964454475931915,-0.3033428178887644,0.8664953213479795), 0.7486828364294617, 0.8299580318606381, 0.8867745025183218,  (gDx*2, gDy*0, gDz*0)), rgb(0.2816728386030699,0.15311413978735772,0.46711648771104586)  );
  draw(  ellipsoid( (0.8272681445490871,0.5582502364302713,-0.063119652564451), (-0.40661775214301754,0.6724859129179206,0.618404964864538), (0.38767179502314725,-0.48592115662162455,0.7833206296855926), 0.7618281868563066, 0.8424304882094515, 0.9198444844557353,  (gDx*2, gDy*0, gDz*1)), rgb(0.28024894396113165,0.16573662837986503,0.4765291161233485)  );
  draw(  ellipsoid( (0.8219974897627395,0.569489132883199,0.0015014498649579505), (-0.42990973804623384,0.6187959421338635,0.6574718238310163), (-0.3734939677649079,0.5410856766969582,-0.7534776350473985), 0.7565278953030055, 0.8275411386281122, 0.9254531505218413,  (gDx*2, gDy*0, gDz*2)), rgb(0.278928043030861,0.17482276408564035,0.48293316804154107)  );
  draw(  ellipsoid( (0.7735429739474573,0.6308971081053735,0.06000088700010885), (-0.5112996019275504,0.5653466380779981,0.6472680247645742), (-0.37443822524583836,0.5313680624559405,-0.7598973599615718), 0.7344705715385696, 0.7842798959403553, 0.8939207413698356,  (gDx*2, gDy*0, gDz*3)), rgb(0.2791248085695307,0.17353615947383094,0.4820387792294061)  );
  draw(  ellipsoid( (0.7081143131094683,0.6980976331289859,-0.105989689070276), (0.6172294417091961,-0.6848811712022377,-0.387253918792315), (0.3429313865146528,-0.2088000860927931,0.9158605724621951), 0.7939285307629175, 0.8771958103584085, 0.9626171229548712,  (gDx*2, gDy*1, gDz*0)), rgb(0.27986017748570563,0.16853734545540086,0.4785266108333566)  );
  draw(  ellipsoid( (0.7488530365217572,0.6586412673810794,-0.07355821228652383), (-0.5437241012000137,0.6740406659276911,0.5000332813423444), (-0.37892378055010006,0.33445606823378166,-0.8628765305391145), 0.7964389302485018, 0.870627785059389, 0.9700025433663133,  (gDx*2, gDy*1, gDz*1)), rgb(0.27939344136077715,0.17177963142304675,0.48081772108737664)  );
  draw(  ellipsoid( (0.7508734996336123,0.6542304189304717,-0.09039660665108663), (-0.49595478935515785,0.6489400515267901,0.5769797712572667), (-0.4361396960420536,0.38840619005168386,-0.8117405971527357), 0.7768214021718767, 0.8253765060183885, 0.9353203053309548,  (gDx*2, gDy*1, gDz*2)), rgb(0.2801780641499513,0.16624725444983693,0.4768932988829962)  );
  draw(  ellipsoid( (0.6654139638568008,0.7264100269801661,-0.17190907307950543), (-0.5109110178151837,0.6110914804648303,0.6045966708296987), (-0.544237253943011,0.3144768277727006,-0.7777596905308152), 0.7288390190089553, 0.7461610360560866, 0.855856525198746,  (gDx*2, gDy*1, gDz*3)), rgb(0.2814732641068035,0.1551951765873732,0.4687114030538398)  );
  draw(  ellipsoid( (0.2869033131223248,0.9576246459812434,0.02532837003700838), (-0.86987302018101,0.24935603510979618,0.42560838398167167), (0.40125729610887323,-0.14414092119733707,0.9045529156195227), 0.7594258920723079, 0.7965976579328041, 0.8775801284110184,  (gDx*2, gDy*2, gDz*0)), rgb(0.28286782048787024,0.13623032428245882,0.4536805410138347)  );
  draw(  ellipsoid( (0.2373900264926394,0.9701853291973509,0.04885081710730807), (-0.8273710282319243,0.1755823868715894,0.5335053955330947), (0.5090217647255092,-0.16706661075961127,0.844384740864628), 0.7464766478703642, 0.7728197825622991, 0.8686278479737677,  (gDx*2, gDy*2, gDz*1)), rgb(0.2823753422301668,0.14463417309425958,0.4604866620869191)  );
  draw(  ellipsoid( (-0.13921524209915825,0.967309310429127,0.21197125825073407), (-0.7271529805216685,-0.24516235365838215,0.6412050867445893), (0.6722110228832356,-0.06487001083559046,0.737512184583051), 0.7001432400158795, 0.7132638682901139, 0.8150413688953768,  (gDx*2, gDy*2, gDz*2)), rgb(0.2819851542613085,0.14967075552247697,0.46445633784665885)  );
  draw(  ellipsoid( (-0.33031862831862446,-0.6833001574051186,0.6511455280318239), (0.49726348476678,-0.7123472299066104,-0.49526805950109193), (0.8022584561743704,0.16019462829498224,0.5750817772822819), 0.6132842794168989, 0.615126096126498, 0.7479210777904018,  (gDx*2, gDy*2, gDz*3)), rgb(0.2758617252306823,0.19167441815610547,0.49399616790332035)  );
  draw(  ellipsoid( (0.3998986414825274,-0.8626623293242369,-0.3096688264990696), (0.03657876337145383,-0.3225699306242036,0.9458385876709152), (0.9158291710852215,0.3895668689993773,0.09744015582591323), 0.5661206383664439, 0.6117499225308209, 0.6878249751351567,  (gDx*2, gDy*3, gDz*0)), rgb(0.2795371540349651,0.17083992729276146,0.48016448165924946)  );
  draw(  ellipsoid( (-0.42875917830242205,0.8603466750271254,0.27562504565297063), (-0.1827110512414012,-0.38136617624167474,0.9061879006988927), (-0.8847498170532134,-0.33817663785154084,-0.3207090937839671), 0.548822815035203, 0.5753921910231126, 0.690231678134027,  (gDx*2, gDy*3, gDz*1)), rgb(0.2738243569116591,0.20101973723792174,0.49967146085404457)  );
  draw(  ellipsoid( (0.07437702365278137,-0.7967366219996376,0.5997322848714752), (0.5013799190242954,-0.4899832588253682,-0.7131161075659863), (0.8620244979997637,0.3537331780247302,0.3630242465907475), 0.49761645185478337, 0.5068854793689622, 0.673127555299072,  (gDx*2, gDy*3, gDz*2)), rgb(0.2568690319870408,0.257365195422339,0.5270354156557339)  );
  draw(  ellipsoid( (-0.046812022883836235,-0.638062772455176,0.7685600385918706), (0.5352058763088691,-0.6656664327900872,-0.5200412197326054), (0.8434235617280199,0.38699366747187935,0.3726561375567098), 0.39551805310793214, 0.43858364683338946, 0.6415075940057606,  (gDx*2, gDy*3, gDz*3)), rgb(0.22291000543696998,0.3371786499711498,0.5484194675617551)  );
  draw(  ellipsoid( (0.7302409041895597,0.6684180393746046,-0.14129949216831533), (-0.6267004414799779,0.7377267061746057,0.25100968835000703), (-0.2720198126924168,0.09474508765826746,-0.9576160973309762), 0.6581694149562998, 0.6995158803587349, 0.7669741164619533,  (gDx*3, gDy*0, gDz*0)), rgb(0.282607153393209,0.14116327081519453,0.45770701651926826)  );
  draw(  ellipsoid( (-0.76335725206976,-0.6459754749396266,-0.0011796139320082273), (-0.6015267068294696,0.7101635445106729,0.3658324220444805), (0.23548105376739062,-0.27997040169375276,0.930679991990823), 0.6689566867857137, 0.7164852327912811, 0.7955557211040102,  (gDx*3, gDy*0, gDz*1)), rgb(0.2814022053219777,0.1559228903040369,0.46926758220464415)  );
  draw(  ellipsoid( (-0.7537007012146492,-0.6454465953202776,-0.12383030960959467), (-0.6259711336937742,0.6476036002514838,0.4344763707308459), (0.20023833991190595,-0.404979344573004,0.8921302246300953), 0.6670900335730229, 0.71925772907341, 0.8131184399739235,  (gDx*3, gDy*0, gDz*2)), rgb(0.2791468844185554,0.17339181057332315,0.48193843446111195)  );
  draw(  ellipsoid( (-0.6971053895648692,-0.6805458462807592,-0.22561344585289536), (-0.6924374494601268,0.5574419388072575,0.4580271426934875), (0.1859420727666732,-0.47551638874891694,0.8598312099513612), 0.6558966010462532, 0.7043016625380477, 0.8101545545140034,  (gDx*3, gDy*0, gDz*3)), rgb(0.2774490738976494,0.1834836102318066,0.4887493080337405)  );
  draw(  ellipsoid( (0.6912021663398028,0.7140449547536476,-0.11126260754640276), (0.6723425392437541,-0.6918521733712132,-0.2632414863291686), (0.264943532041121,-0.10714650154060096,0.9582925190343425), 0.7219619869299795, 0.7954465736175524, 0.8862328832976311,  (gDx*3, gDy*1, gDz*0)), rgb(0.2785825516053146,0.1769485968315489,0.48438395295142744)  );
  draw(  ellipsoid( (0.7337389747535995,0.6790702337077077,-0.022152530774064226), (0.6214780145167497,-0.6839727791176117,-0.3820292068657178), (0.27457639082749824,-0.2665424077295961,0.923884706271218), 0.7258165585504528, 0.7983378690725671, 0.8975310214557997,  (gDx*3, gDy*1, gDz*1)), rgb(0.27771190062870665,0.18202848410462044,0.48779109804955567)  );
  draw(  ellipsoid( (0.7135508318368674,0.6982988067592353,0.05678016258820385), (-0.6414986464951784,0.6186241253369591,0.4536338590713179), (-0.2816464040774688,0.3601152149378161,-0.8893774986137899), 0.7076352306099556, 0.771083583753838, 0.8792135045162454,  (gDx*3, gDy*1, gDz*2)), rgb(0.276993831458025,0.18595061356852235,0.4903607053039878)  );
  draw(  ellipsoid( (0.6233579186323096,0.7702558422058042,0.13465081814167393), (0.7219291958854294,-0.5007662820284101,-0.47754724050261294), (-0.300404962353509,0.39489121075464545,-0.86822680807616), 0.6734016290866722, 0.7198813384174301, 0.8246894570937826,  (gDx*3, gDy*1, gDz*3)), rgb(0.27841986009862335,0.17792334680468538,0.4850435131136891)  );
  draw(  ellipsoid( (0.6015250775053874,0.793703573481917,-0.09056610057948218), (0.7512954429215524,-0.6005998993105146,-0.27355971632078446), (0.2715193152934893,-0.0965111309136722,0.9575817787700008), 0.7190175820198432, 0.7931357082165382, 0.8972075727692213,  (gDx*3, gDy*2, gDz*0)), rgb(0.27658711908796696,0.18804734563799158,0.49170328879791325)  );
  draw(  ellipsoid( (0.6330636362015498,0.7726947041493214,-0.046618952142497116), (0.707475940715368,-0.6019705797515578,-0.3702826142589985), (-0.31417865272512313,0.20143067118283903,-0.9277485968077007), 0.7148517775543198, 0.7783240268837213, 0.8846989034344606,  (gDx*3, gDy*2, gDz*1)), rgb(0.27748613558596924,0.18327842017836382,0.4886141890538866)  );
  draw(  ellipsoid( (0.5672107870371621,0.8231626271970554,-0.025984846636545468), (0.7238040989103987,-0.5133026085084655,-0.4611161008888228), (-0.39291163061073825,0.2427420879955669,-0.8869592601953955), 0.6869594685248914, 0.7299669676266299, 0.8312345935505367,  (gDx*3, gDy*2, gDz*2)), rgb(0.2796631259489313,0.16995692671739068,0.4795390679951386)  );
  draw(  ellipsoid( (0.3040589902778418,0.9525950759769227,-0.01052386125620857), (-0.7789515088345635,0.2549629578434894,0.5729122419813143), (0.5484365754721156,-0.1660015402103644,0.8195491512607436), 0.6320551325661392, 0.6583955741692349, 0.7437681185292917,  (gDx*3, gDy*2, gDz*3)), rgb(0.28175838666910086,0.15222209858514435,0.4664328235875647)  );
  draw(  ellipsoid( (0.15191916185508936,0.9883928706841312,0.00031849965377434903), (-0.9133724465426125,0.1402651215310738,0.3821995154086878), (0.377718601816231,-0.05835433885029973,0.9240797741425445), 0.6425142597387016, 0.6868232047756572, 0.7543073295540637,  (gDx*3, gDy*3, gDz*0)), rgb(0.28225469043801615,0.14634736777542848,0.46185126735465765)  );
  draw(  ellipsoid( (0.12132328412167533,0.9922616911287725,-0.026408276130792045), (-0.8476545927445326,0.11741228225959657,0.5173935130765491), (0.5164904182354516,-0.04038678364096539,0.8553400233697159), 0.625115774545027, 0.6608107807055121, 0.73522198257079,  (gDx*3, gDy*3, gDz*1)), rgb(0.28203646313702546,0.14903811581171317,0.4639604369262674)  );
  draw(  ellipsoid( (-0.01839675864214028,-0.9907862367651044,0.1341797015509631), (0.706535101889009,-0.10783862474424884,-0.6994133118635625), (0.7074388376885676,0.08193573121729317,0.7020091358947801), 0.5828100407148735, 0.6076235305148396, 0.6883380530765478,  (gDx*3, gDy*3, gDz*2)), rgb(0.28149724156689065,0.15494515477724335,0.46851978528859595)  );
  draw(  ellipsoid( (-0.07649241547655072,-0.8508426646558173,0.5198227297607823), (0.5825235894344802,-0.4612551720107038,-0.6692607369670993), (-0.809206511421486,-0.25161563205565063,-0.5309184453174552), 0.5025944203129098, 0.5312400339382204, 0.6404967820421033,  (gDx*3, gDy*3, gDz*3)), rgb(0.27201980844863693,0.20852420559437157,0.5039922543963057)  );
