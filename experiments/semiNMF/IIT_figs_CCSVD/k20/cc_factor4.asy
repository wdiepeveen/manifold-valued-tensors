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

  draw(  ellipsoid( (0.9614229881652457,-0.25897269325159794,-0.09273069598263192), (0.23460124345957095,0.5959509248617108,0.7679874684678747), (0.1436248390586821,0.7601155433922432,-0.6337162348392872), 0.5478279600150029, 0.8402197840988004, 2.2147252601340393,  (gDx*0, gDy*0, gDz*0)), rgb(0.1272628455295434,0.5679572413176341,0.5503603179853127)  );
  draw(  ellipsoid( (-0.9558780403661269,0.24291730793996155,0.16519186859229817), (0.2773324283328328,0.5607852416582132,0.7801324483274321), (-0.09687051223580749,-0.7915245379907944,0.6034111447575622), 0.5683878787702508, 0.9356166435593329, 2.0286551248780826,  (gDx*0, gDy*0, gDz*1)), rgb(0.13606014401116195,0.5421268822942802,0.5543653199560861)  );
  draw(  ellipsoid( (-0.9553026610746763,0.21205580804943544,0.20598339742839267), (0.29135900680903026,0.5573394439705471,0.7774848380167018), (-0.05006750338191147,-0.8027484527964549,0.5942122235683143), 0.5883275798862091, 1.0300896393894596, 1.592719170430077,  (gDx*0, gDy*0, gDz*2)), rgb(0.1584154697995248,0.4842005966489801,0.5580411148040761)  );
  draw(  ellipsoid( (-0.9628811046204401,0.14649280076675247,0.22671532301199618), (0.26042003202302755,0.7251030110599284,0.6375006119784911), (0.07100271323017478,-0.6728785051369508,0.7363376481198118), 0.6162350454967037, 1.0855292212366816, 1.2700288351988194,  (gDx*0, gDy*0, gDz*3)), rgb(0.1835771776065639,0.4231256589022233,0.5569783877839493)  );
  draw(  ellipsoid( (-0.9113553342702383,0.39812060852434117,0.10455350671015545), (0.32153136712246927,0.5299532822490908,0.7847082888499213), (-0.2570000674193259,-0.7487653168394113,0.6109840142301908), 0.4821222098603397, 0.7076746347923569, 1.5771502793364975,  (gDx*0, gDy*1, gDz*0)), rgb(0.14116635986786558,0.528474320261703,0.5558085921916593)  );
  draw(  ellipsoid( (0.9010780340395295,-0.3953373247038039,-0.17823236593698577), (0.3141898109892672,0.3118687863372781,0.896673085789376), (-0.29890322721908696,-0.8639712146851602,0.40522907219677967), 0.5312617711590097, 0.9236783655957717, 1.414651081905449,  (gDx*0, gDy*1, gDz*1)), rgb(0.15991519159801038,0.4804244606812751,0.5580935914664287)  );
  draw(  ellipsoid( (-0.9214597169884207,0.3276490405037607,0.20870576471335123), (0.12387948358775218,-0.26136303634168756,0.9572581871053835), (0.3681926988869435,0.9079292205294522,0.20024651556301773), 0.60859420263385, 1.1100495950979847, 1.4288182844349424,  (gDx*0, gDy*1, gDz*2)), rgb(0.1702646366990772,0.45475301165950577,0.5580037180487321)  );
  draw(  ellipsoid( (-0.9478887477373678,0.2087105340640229,0.2407214882049274), (0.0643713923597188,-0.6145162674531129,0.7862735407484887), (0.31203084103747797,0.7607954192857663,0.5690580675425304), 0.7002715825567492, 1.2327741879689569, 1.5990367323414054,  (gDx*0, gDy*1, gDz*3)), rgb(0.17376525620437588,0.44626989260591876,0.557822426477809)  );
  draw(  ellipsoid( (-0.881069889814048,0.4728122638073792,-0.012822342079457113), (0.2723111841980457,0.5292356085234659,0.8035896276280916), (-0.3867330710240847,-0.7045269575141452,0.5950456267481867), 0.4588778744161257, 0.6390007961879914, 1.206778080528565,  (gDx*0, gDy*2, gDz*0)), rgb(0.1598524875288042,0.4805820520775057,0.5580918011395036)  );
  draw(  ellipsoid( (-0.8823561255693719,0.4702391190457845,0.017967709627307444), (0.09901640651251588,0.14819632871790983,0.9839886175133741), (0.4600471920483319,0.8700074821940198,-0.17732332620066485), 0.5277666331704225, 0.8928716205686019, 1.2125495470514518,  (gDx*0, gDy*2, gDz*1)), rgb(0.17408020843795924,0.44551096912087895,0.5578035901062828)  );
  draw(  ellipsoid( (-0.9171366140361616,0.3910567116263585,0.07703946706892238), (-0.07944949026969218,-0.3687789387926836,0.9261154748727765), (0.3905742051008625,0.8432536644420355,0.36928992365742686), 0.63606212656573, 1.0811953813423683, 1.555367501890431,  (gDx*0, gDy*2, gDz*2)), rgb(0.16793622034499803,0.4604558464656112,0.5580866533116203)  );
  draw(  ellipsoid( (-0.9536107264922905,0.25258334782477215,0.1637932682391923), (-0.003745133390833645,-0.5539985357856508,0.8325092169599325), (0.30101919589077825,0.7932762915586131,0.5292449045161263), 0.7659372716848962, 1.3663215014936647, 1.9695515613860755,  (gDx*0, gDy*2, gDz*3)), rgb(0.16270161445320439,0.4734399072494118,0.558143018823753)  );
  draw(  ellipsoid( (0.8762243152219295,-0.43262062678106544,0.21229776894121796), (0.1389651676387452,0.6486599652682846,0.7482839913036246), (0.461432152755956,0.626162632821303,-0.628490831803693), 0.49393788778674497, 0.6540419766811618, 1.1088341461840407,  (gDx*0, gDy*3, gDz*0)), rgb(0.17635877660529045,0.44005839863103163,0.5576457249098389)  );
  draw(  ellipsoid( (-0.8780220984643923,0.45577807828428557,-0.14609427765545333), (-0.014770544577499685,0.27929199998348736,0.9600926047825323), (0.4783921253752713,0.8451404156119309,-0.23849245748707348), 0.5535047254177159, 0.9209169683988309, 1.1575391657196183,  (gDx*0, gDy*3, gDz*1)), rgb(0.1842642173091235,0.4215407885405983,0.5569018121193954)  );
  draw(  ellipsoid( (-0.9238096865540028,0.3816041340406282,-0.030886047207391975), (-0.12349595111498732,-0.2206592269112673,0.9675010365044197), (0.3623871039195514,0.8976011310498799,0.25097369673090486), 0.6513959154791287, 1.1439392239604411, 1.5186655481311795,  (gDx*0, gDy*3, gDz*2)), rgb(0.1718698495900348,0.45084866259420603,0.5579290259447811)  );
  draw(  ellipsoid( (-0.9577442764574409,0.28134873360218154,0.05973935899774785), (-0.023792430691428536,-0.28448796834790746,0.958384325887623), (0.2866353452643387,0.9164657582068896,0.27916069365056473), 0.7574735996494784, 1.45274168077203, 1.9364565164830394,  (gDx*0, gDy*3, gDz*3)), rgb(0.16180886306304157,0.47567298506473943,0.5581343612569911)  );
  draw(  ellipsoid( (-0.8557871408188583,0.4215587851904338,0.2998609014824193), (0.5033802891399671,0.54487177178413,0.6706139253088814), (-0.11931745100548828,-0.7248468410233908,0.6784986388666996), 0.7108556937918706, 0.9222967848406919, 1.9141615389389113,  (gDx*1, gDy*0, gDz*0)), rgb(0.15586535404669716,0.49064993740325374,0.5579093017102976)  );
  draw(  ellipsoid( (-0.9019599230838624,0.33085575849060733,0.27749371925172994), (0.43113840070655074,0.6538966867530522,0.6217224481145734), (-0.024248228528096695,-0.6804069298052347,0.7324332278687687), 0.67743186544451, 0.8992939634264268, 1.7409846962388977,  (gDx*1, gDy*0, gDz*1)), rgb(0.1611630429619394,0.477291689204401,0.5581234299756591)  );
  draw(  ellipsoid( (-0.9134473447932939,0.27678844024197535,0.29833221019275735), (0.39684591320311136,0.7682340286027091,0.5023243956556765), (0.09015136972930451,-0.5772388038284209,0.8115836949388918), 0.6316604313315962, 0.8757199898448792, 1.4200973643395842,  (gDx*1, gDy*0, gDz*2)), rgb(0.17704318478061865,0.4384304416375044,0.5575938102568431)  );
  draw(  ellipsoid( (0.9318706719154403,-0.1795910393199901,-0.3152207312659365), (0.26673626578810505,0.9281006994868594,0.2597707761189079), (0.24590407751251986,-0.32615356844279386,0.9127732656332287), 0.6002844181784293, 0.833545814936793, 1.165339812451021,  (gDx*1, gDy*0, gDz*3)), rgb(0.196887050898573,0.39317476027087417,0.5550984636115642)  );
  draw(  ellipsoid( (-0.7899649576629323,0.5235731740065412,0.31910264355739487), (0.5492478239966018,0.37291844558781356,0.7478359852102378), (-0.2725475985944473,-0.7660306550008849,0.5821639306065906), 0.6030262152580766, 0.900342883716101, 1.5762817601342696,  (gDx*1, gDy*1, gDz*0)), rgb(0.16148762394630767,0.4764781490526534,0.558128923898888)  );
  draw(  ellipsoid( (0.8268825149812374,-0.4666465124987313,-0.31385719490092184), (-0.5064674564747861,-0.37533024020216754,-0.7762846941179145), (-0.2444504488649284,-0.8008546954114433,0.5466952852249408), 0.6009620988486355, 1.0352940142103135, 1.3050292898348226,  (gDx*1, gDy*1, gDz*1)), rgb(0.1792638305692995,0.4331810283338376,0.5574074938358656)  );
  draw(  ellipsoid( (0.8626929598890916,-0.4138034465092033,-0.2907362457880051), (-0.18102142110094255,-0.7894623804772769,0.5864984185091606), (-0.4722203956421132,-0.4533385682558422,-0.755970925678593), 0.5989655464236472, 1.0099856753913592, 1.1392097908869374,  (gDx*1, gDy*1, gDz*2)), rgb(0.19394624954272588,0.3996649029101517,0.5555889049764322)  );
  draw(  ellipsoid( (0.9116421591339355,-0.2873593599741927,-0.29382507027964544), (-0.13384080899272874,-0.8835370145214805,0.4488306827954623), (0.3885810231313268,0.3698471876406411,0.8439300008036921), 0.6174944430117236, 0.870604126004512, 1.2030739658926497,  (gDx*1, gDy*1, gDz*3)), rgb(0.19634053377983396,0.39437642094017367,0.5551929347576526)  );
  draw(  ellipsoid( (-0.7612934588352045,0.6115654550421126,0.21545292695590046), (0.39586857673036724,0.1752142683866387,0.9014366478634028), (-0.5135370867738802,-0.7715488671191665,0.37548902268232437), 0.49852699591319893, 0.7403487699318817, 1.2121190318704862,  (gDx*1, gDy*2, gDz*0)), rgb(0.16887005072037425,0.45816409672097724,0.5580565456332504)  );
  draw(  ellipsoid( (0.8062401611540563,-0.5740058476999154,-0.1431575682546441), (0.004022018714107676,-0.23666442657090905,0.9715831269435087), (0.5915747001924142,0.7839051192600828,0.18849970315650563), 0.5415102137982623, 0.852896638287107, 1.2414075190973874,  (gDx*1, gDy*2, gDz*1)), rgb(0.17525673740105774,0.4426896948286083,0.5577248954040119)  );
  draw(  ellipsoid( (0.8546033365530797,-0.5034977368498107,-0.12705575996176738), (-0.18682428349691815,-0.5264116395630717,0.8294501026754905), (0.4845098804443588,0.6851137239317553,0.543938747500264), 0.5988252499601486, 0.8624132305791911, 1.441851379600624,  (gDx*1, gDy*2, gDz*2)), rgb(0.16962863826994085,0.4563045590830266,0.5580305939539232)  );
  draw(  ellipsoid( (0.9184746835902662,-0.3307999471923287,-0.2167386687725859), (-0.10808800857317198,-0.7371460751024993,0.6670327176111125), (0.3804224467699639,0.5892258131637926,0.7128055156177043), 0.6686481054281795, 0.9359874147948543, 1.5641972821950865,  (gDx*1, gDy*2, gDz*3)), rgb(0.17248776609515581,0.4493513263578824,0.5578969887961034)  );
  draw(  ellipsoid( (0.760916119384568,-0.6473646774910203,0.0438820418585676), (0.15556797559851332,0.24767696083798196,0.9562712627901359), (0.6299248083929622,0.7208155779441843,-0.289170604253063), 0.4645571746946931, 0.6435437165418416, 1.067582102104588,  (gDx*1, gDy*3, gDz*0)), rgb(0.17440078482157537,0.4447402666112355,0.5577833427084182)  );
  draw(  ellipsoid( (-0.7828508281583189,0.6155210151589846,-0.09098604700440181), (-0.19150927682140137,-0.09923106946524522,0.9764616693675833), (-0.5920040352423096,-0.7818484985919183,-0.19556110939137478), 0.5108993112294724, 0.8096738223972277, 1.2423974082237608,  (gDx*1, gDy*3, gDz*1)), rgb(0.16901767994385025,0.4578022108744828,0.5580514951598157)  );
  draw(  ellipsoid( (-0.8455502542501379,0.5334549349896508,-0.02169331400949784), (-0.28882709471505114,-0.42287517971965505,0.8589269420244846), (-0.4490252519580497,-0.732531511071201,-0.5116384547624945), 0.6021576714632852, 0.9261763245800687, 1.575842822565411,  (gDx*1, gDy*3, gDz*2)), rgb(0.1616252215554308,0.4761332700079859,0.5581312529037818)  );
  draw(  ellipsoid( (0.9245289230419104,-0.36176752140191143,-0.11987714927242353), (-0.15835509232010356,-0.6507583160212781,0.7425882296841513), (0.3466553550650231,0.6675611392133813,0.6589326143217342), 0.7040061946759272, 1.0945394176502734, 1.850516414248864,  (gDx*1, gDy*3, gDz*3)), rgb(0.16126275844119686,0.47704175914061564,0.5581251177799796)  );
  draw(  ellipsoid( (0.3191459297851432,-0.690067112962083,-0.6495792908565914), (0.9416989165081558,0.15386307180097833,0.29921448123934813), (-0.10653180815578803,-0.7072011982060994,0.6989402257038263), 0.7241021622986599, 0.9330946031308307, 1.2921988876009687,  (gDx*2, gDy*0, gDz*0)), rgb(0.21008132241136723,0.3646297937735356,0.5523111951005685)  );
  draw(  ellipsoid( (-0.31068244458654976,0.7333868282309086,0.6046653444700902), (0.9490931957543826,0.2741211804801535,0.15517630033255836), (0.051947323304077885,-0.6220943164742586,0.7812170869943683), 0.7154350324846322, 0.8756798566063984, 1.2835450139291398,  (gDx*2, gDy*0, gDz*1)), rgb(0.20788173218911266,0.3693425296467139,0.552850490290526)  );
  draw(  ellipsoid( (0.4431400407735756,-0.6976835405069157,-0.5629072584084602), (0.868073926498108,0.4907095978595949,0.07517811318882536), (0.22377346222765318,-0.5219595462046936,0.8230939616644257), 0.6899941220955291, 0.8108476984418624, 1.1852931755900495,  (gDx*2, gDy*0, gDz*2)), rgb(0.2144522249288179,0.3552895781332153,0.5511390715034772)  );
  draw(  ellipsoid( (0.7268101421038768,-0.4086054734173282,-0.5520766110136738), (0.5223884796952992,0.8507272267450762,0.05808151131182676), (-0.4459341808121199,0.3306126929929221,-0.8317679686158823), 0.6512404145445982, 0.7422554506994251, 1.0418249356851836,  (gDx*2, gDy*0, gDz*3)), rgb(0.22757387351661645,0.3270894309548882,0.5466240035636957)  );
  draw(  ellipsoid( (0.49296248599413645,-0.6588731691538303,-0.5682201460452356), (-0.8267159713889395,-0.15117489690392946,-0.5419288266890113), (-0.2711617414992817,-0.7369072516370523,0.6192245250567866), 0.6684028100835628, 0.9054967317131001, 1.256118097760658,  (gDx*2, gDy*1, gDz*0)), rgb(0.20182764024437233,0.38239396736499554,0.5541749165295102)  );
  draw(  ellipsoid( (-0.5496889190575648,0.6792564161268656,0.4862641395536097), (0.8330566193329386,0.4024409640194918,0.3795496534903404), (-0.06211892828197867,-0.6137197989551487,0.7870763921752132), 0.6477899386370445, 0.899022979216683, 1.0876631582708836,  (gDx*2, gDy*1, gDz*1)), rgb(0.22012794120436502,0.34315476281920426,0.5493887287080305)  );
  draw(  ellipsoid( (0.5976072764182033,-0.6992133633512648,-0.392385289840265), (0.6808971297747668,0.7009888496228485,-0.21211725853621638), (0.42337293470001297,-0.1404112004629522,0.8950084094286775), 0.6151251897518342, 0.8103670268593404, 0.970939486850284,  (gDx*2, gDy*1, gDz*2)), rgb(0.23252876505865233,0.3162103624049172,0.5444263360010575)  );
  draw(  ellipsoid( (-0.6687811929602249,0.6717664971384364,0.31853020149595496), (-0.5181822376889559,-0.7284120505843958,0.4482221024303708), (-0.533121828934405,-0.13470581979183568,-0.8352457468481025), 0.6116910269501866, 0.6822963618550291, 0.9717815806166313,  (gDx*2, gDy*1, gDz*3)), rgb(0.2277772825788277,0.3266476797630513,0.5465419685525306)  );
  draw(  ellipsoid( (-0.5927633697260289,0.6404618677568544,0.4883033723623507), (0.5225684885443046,-0.1554890461402029,0.8382990703268918), (-0.6128244139160618,-0.752084937011646,0.24251697925840932), 0.5661843416111761, 0.7611599192188204, 1.1076935557131016,  (gDx*2, gDy*2, gDz*0)), rgb(0.19551665155946574,0.39619187808731726,0.5553323793746104)  );
  draw(  ellipsoid( (0.6660464914983889,-0.6777972672765417,-0.3114047777975462), (0.07816650946256079,-0.35176050334857517,0.9328207464901263), (0.7418032542130873,0.645643409907055,0.18130780259675633), 0.5712668561297016, 0.7645815018899201, 1.0201682049011342,  (gDx*2, gDy*2, gDz*1)), rgb(0.21021592189478006,0.36434162200718634,0.5522776168251852)  );
  draw(  ellipsoid( (-0.635505418367545,0.7686393581898971,0.07298219143677649), (-0.38225412294099753,-0.39534831326296355,0.8352134437943025), (-0.6708313116723906,-0.5028849254467194,-0.5450615589438935), 0.5688377103273814, 0.7018353611908462, 1.0063446933055502,  (gDx*2, gDy*2, gDz*2)), rgb(0.21069940346923768,0.36330720315560405,0.5521545115640899)  );
  draw(  ellipsoid( (0.5793114554412016,-0.8139565217321804,0.04327838171936527), (0.6283134135329482,0.41210081060905157,-0.659844812262654), (0.5192499320839351,0.4094480463074809,0.7501545210193803), 0.5938395043987245, 0.6716715324294098, 1.0136082021575437,  (gDx*2, gDy*2, gDz*3)), rgb(0.21366854853863892,0.3569632410609414,0.5513579839562144)  );
  draw(  ellipsoid( (0.682597597362844,-0.7024149985313026,-0.20167719234643938), (0.18966781221301895,-0.09623398608373875,0.9771208425434194), (0.7057525353017164,0.7052320112989088,-0.06753642835163363), 0.495451888711301, 0.5732234580916121, 1.046162681325418,  (gDx*2, gDy*3, gDz*0)), rgb(0.17971090868439565,0.43213108898403735,0.5573663959718693)  );
  draw(  ellipsoid( (-0.6667333977563686,0.7283009009438307,-0.15825414370768498), (-0.2911746744273262,-0.059079805039833094,0.9548439064101504), (-0.686064053342244,-0.6827059208186635,-0.25145325687043973), 0.5222217007719264, 0.6482346916389966, 1.1471045244345104,  (gDx*2, gDy*3, gDz*1)), rgb(0.17728098746615215,0.43786479593228167,0.5575757721264613)  );
  draw(  ellipsoid( (-0.6448021821308333,0.7385559322778913,-0.1968890063372016), (-0.48602478317176256,-0.19736472287145435,0.8513677679526722), (-0.5899237713994552,-0.644656731185682,-0.4862176908287423), 0.5631523181999699, 0.7141591372621399, 1.2112595465882516,  (gDx*2, gDy*3, gDz*2)), rgb(0.18071018928998436,0.4297858244632264,0.5572739160018578)  );
  draw(  ellipsoid( (-0.7419280809294948,0.6686631473069617,-0.04931853770972256), (-0.486188502063945,-0.4858891097657566,0.7263143351689115), (-0.46169628890548214,-0.5628511068165266,-0.6855911087275831), 0.6339190066906607, 0.7639690130831943, 1.2028798995023062,  (gDx*2, gDy*3, gDz*3)), rgb(0.1975546571490009,0.3917068535096574,0.5549830609545364)  );
  draw(  ellipsoid( (0.3308286544194037,-0.219864765606915,-0.9177210285591494), (0.21781478023525724,-0.9284316871429299,0.3009506999825705), (0.9182097379826213,0.29945631927469557,0.25926201017568606), 0.6979187136785933, 0.8860795069625211, 1.0503585084530767,  (gDx*3, gDy*0, gDz*0)), rgb(0.2423847998151298,0.29379721503845135,0.5389790825269457)  );
  draw(  ellipsoid( (-0.3494427330304097,0.3539356552710776,0.867536355584059), (-0.22806633879011387,-0.9301900468788188,0.28763209452085453), (0.9087769371210463,-0.09734489516874673,0.40576896128397455), 0.7360577690198724, 0.9421716286368715, 1.0244323298962605,  (gDx*3, gDy*0, gDz*1)), rgb(0.2558764523563672,0.260071854129,0.5280706105120536)  );
  draw(  ellipsoid( (0.38366840218752385,-0.3197815265053778,-0.866336154427808), (0.5054026434322042,0.8578752684597679,-0.09283421662739039), (0.7728950285611966,-0.40223102698694346,0.49075806234275543), 0.7193205549833295, 0.8980210313796557, 1.013301294291788,  (gDx*3, gDy*0, gDz*2)), rgb(0.25511401537384343,0.2621161643471571,0.5288308744504723)  );
  draw(  ellipsoid( (0.4910514582312089,-0.0628427805630759,-0.868860892375819), (0.29786013488061036,0.9493916849805869,0.09967330905827602), (0.8186255587372702,-0.3077437463553453,0.484918530437092), 0.6375933292057263, 0.8356218757768311, 0.9478472826803832,  (gDx*3, gDy*0, gDz*3)), rgb(0.24350797212142994,0.2911535428400452,0.538249144171858)  );
  draw(  ellipsoid( (0.34854415205725864,-0.5093583972582947,-0.7868106488915536), (0.3645086501176695,-0.6997094556302323,0.6144429360738378), (-0.8635105199452301,-0.5009597796670954,-0.058214097088364815), 0.6880545057112049, 0.8733562757149869, 0.9492975320425941,  (gDx*3, gDy*1, gDz*0)), rgb(0.2577695991702392,0.2548787382903782,0.5260652740815417)  );
  draw(  ellipsoid( (0.3511154654294264,-0.6288449507599189,-0.6937376722076093), (0.28678697141865783,0.7775268519313555,-0.5596474136009474), (0.8913311185865727,-0.0024540638685786043,0.45334624142010693), 0.6881436190508818, 0.8690559449755496, 0.9166689979251271,  (gDx*3, gDy*1, gDz*1)), rgb(0.2632963958627737,0.23875004410412348,0.5192608220227832)  );
  draw(  ellipsoid( (-0.4435846797796717,0.5936955297949745,0.6713853213812696), (-0.29224675050257903,-0.8039917731515734,0.5178697379894601), (-0.8472452234657032,-0.033508903384831444,-0.5301440226094196), 0.67979758187672, 0.7793812107303952, 0.9146436096917514,  (gDx*3, gDy*1, gDz*2)), rgb(0.2647543478409531,0.23418832040725002,0.517169162361686)  );
  draw(  ellipsoid( (-0.5278773747458642,-0.18715090947770463,0.8284443338650357), (0.24945674262051032,-0.966562387723392,-0.05940104544214534), (-0.8118600931149278,-0.1753045570432311,-0.5569124720162922), 0.6351892863850032, 0.6934511757938915, 0.8985238891406562,  (gDx*3, gDy*1, gDz*3)), rgb(0.25251333685213256,0.2689522839728268,0.5312862592670858)  );
  draw(  ellipsoid( (-0.3875855964380833,0.5442743016409026,0.7440046303666683), (-0.47313111664035534,0.5752256104272505,-0.6672798839881565), (-0.7911538105200441,-0.6106398133776613,0.03456394679010346), 0.6077259628328769, 0.7320561678214318, 0.9209960736923971,  (gDx*3, gDy*2, gDz*0)), rgb(0.2412123834730025,0.2965418874791006,0.5397237751232904)  );
  draw(  ellipsoid( (0.4460650680817278,-0.7407721326685133,-0.5022774158759143), (0.0824550491338919,-0.5248030831565497,0.8472206848169531), (0.8911942099843169,0.41933086145801535,0.17301592042152675), 0.5984853122579794, 0.6838417366219194, 0.8621625357789267,  (gDx*3, gDy*2, gDz*1)), rgb(0.25074966159223677,0.27346283787664716,0.532823302845354)  );
  draw(  ellipsoid( (0.33346169590753405,-0.9402085714816134,0.06936237795069594), (-0.37159208198261906,-0.06346100887361067,0.9262246082676523), (-0.8664425093278227,-0.33463493909818753,-0.37053587621775397), 0.5762422796640762, 0.6338116961513224, 0.8407234315971976,  (gDx*3, gDy*2, gDz*2)), rgb(0.2459973323452616,0.285200598179298,0.5365245930595975)  );
  draw(  ellipsoid( (0.037640768755422244,-0.8738437585672248,0.4847476231407499), (0.6416816275832034,-0.3507300517152887,-0.6820799950489401), (0.7660469054500979,0.33672765915038255,0.5475277364786488), 0.5567536320824877, 0.6413681339719282, 0.8420275127395997,  (gDx*3, gDy*2, gDz*3)), rgb(0.240394133734277,0.2984328189218453,0.5402148980494953)  );
  draw(  ellipsoid( (-0.2763668445935683,0.22940733009825523,0.9332704024593219), (0.5681779579527368,-0.744205614170796,0.3511862923482728), (-0.7751097827443882,-0.6273199189678731,-0.07532956896258536), 0.5271377558174529, 0.5614322252050948, 0.9506675364633104,  (gDx*3, gDy*3, gDz*0)), rgb(0.19925995918750433,0.38797864438450147,0.5546729940611406)  );
  draw(  ellipsoid( (0.3149093321492025,-0.7084770497427977,0.6315792765069832), (-0.5005074495764155,0.441432208986966,0.744734783521892), (-0.8064269374492986,-0.5506340661969876,-0.21558691912895825), 0.5162944997005516, 0.5706994532376254, 0.9220688084487018,  (gDx*3, gDy*3, gDz*1)), rgb(0.20358276903624503,0.37859454180766694,0.5538142665966261)  );
  draw(  ellipsoid( (0.27853110439513734,-0.8043268590637199,0.5248606745347912), (-0.5387773555770897,0.3215450010178573,0.7786705166100644), (-0.7950719369980197,-0.49966700522359575,-0.3437928138997383), 0.4982159951571777, 0.6166166847554102, 0.8852599557658745,  (gDx*3, gDy*3, gDz*2)), rgb(0.2099892693671382,0.36482687542407916,0.5523341593972405)  );
  draw(  ellipsoid( (0.22696800164412628,-0.8478950067733255,0.47912376659745665), (0.6688221102276741,-0.2219051154404635,-0.7095316093113506), (0.7079283234200147,0.4814895401046811,0.5167255670755126), 0.5370182556789289, 0.664689078497294, 0.8345746909194806,  (gDx*3, gDy*3, gDz*3)), rgb(0.2362685031897407,0.30785006720854907,0.5425473520819296)  );