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

  draw(  ellipsoid( (0.9194241177965911,0.3270694516923933,-0.21836864560553923), (0.23619887231718018,-0.9032203677900763,-0.35833372702726657), (0.31443502401935636,-0.2778822430073538,0.907696025490444), 0.5375946127578869, 0.7975124228909773, 0.8815117695704469,  (gDx*0, gDy*0, gDz*0)), rgb(0.22037741446311496,0.34261940250306694,0.5493033757970399)  );
  draw(  ellipsoid( (0.9214079173764858,0.341032629197828,-0.1862906213912661), (-0.2445700824120054,0.8814641199839364,0.40398821761275394), (0.3019816626633346,-0.3266768296089593,0.8955943972645916), 0.5525270127314558, 0.810871424669889, 0.920176694167841,  (gDx*0, gDy*0, gDz*1)), rgb(0.21884032985131746,0.3459114444023432,0.5498076127882339)  );
  draw(  ellipsoid( (0.9214286016646036,0.3541928916498335,-0.15973956159681504), (-0.26171205301359013,0.8696356302152422,0.4186175724543232), (0.2871865828025826,-0.3439204358058231,0.8940036915434456), 0.5613552794796208, 0.8144245636540598, 0.9365469804967441,  (gDx*0, gDy*0, gDz*2)), rgb(0.21922447060802247,0.34508913998023216,0.5496830212091262)  );
  draw(  ellipsoid( (0.9221539158000353,0.3579530698248121,-0.1466347686521003), (-0.26268498357643455,0.8577458136097249,0.4418922025092088), (0.28395202936078806,-0.36897387311025737,0.88500255704936), 0.562035147699355, 0.807520261409172, 0.9253064247847651,  (gDx*0, gDy*0, gDz*3)), rgb(0.22177763007826115,0.33961459197491156,0.5488243165208434)  );
  draw(  ellipsoid( (0.918313072534016,0.33287661140004504,-0.21422946201668563), (-0.2204294472943036,0.8795179511305332,0.4217333664825168), (0.32880383145374525,-0.34006078165817977,0.8810486395196153), 0.5706403012894544, 0.8645409044062631, 0.9632403323894773,  (gDx*0, gDy*1, gDz*0)), rgb(0.2149412420630395,0.35424504571890525,0.5509966120140622)  );
  draw(  ellipsoid( (0.9103658006523502,0.36098817547157025,-0.20229099379930976), (0.2603792434410291,-0.8796656135570275,-0.3979837407613727), (0.32161585562232875,-0.3096384108844464,0.8948113186126291), 0.5800009733951065, 0.8812632824836506, 0.9991355525592795,  (gDx*0, gDy*1, gDz*1)), rgb(0.2119980407216662,0.36053146898603694,0.5518140654344892)  );
  draw(  ellipsoid( (0.9051589796823479,0.37839911908176566,-0.19365259662228823), (0.2798044146941394,-0.8733546577097246,-0.3986992994405681), (0.3199948609244865,-0.3067013996297941,0.8964025549088109), 0.5806088724169438, 0.8746727015423464, 1.0000000000000002,  (gDx*0, gDy*1, gDz*2)), rgb(0.21251952711607944,0.3594170629533648,0.5516755799983931)  );
  draw(  ellipsoid( (0.8997902155827452,0.3891496973884592,-0.1973324123503984), (0.29376950926121476,-0.8747126102723279,-0.38545729317136485), (0.3226097384921638,-0.28886045498309565,0.9013781638008415), 0.5711493916534325, 0.8438639586481393, 0.958962479408686,  (gDx*0, gDy*1, gDz*3)), rgb(0.21725797990639978,0.3492954887729384,0.5503055103307972)  );
  draw(  ellipsoid( (-0.9271995664758137,-0.3231917790144583,0.1893357808353526), (-0.16435581931756904,0.8052589875652141,0.5696886233741176), (-0.3365830188579241,0.4970966072333585,-0.7997542338078433), 0.5556330717848673, 0.8185261268678757, 0.9255276687896512,  (gDx*0, gDy*2, gDz*0)), rgb(0.21858489257120453,0.34645824190128766,0.5498904608941345)  );
  draw(  ellipsoid( (0.9123954965943923,0.35288973935172047,-0.20737234544303812), (-0.22452063244534312,0.8551085792665497,0.4673112488171945), (0.3422352164821949,-0.3798133087781639,0.8594282442847501), 0.5589770593865409, 0.8418404936687387, 0.9366914767130923,  (gDx*0, gDy*2, gDz*1)), rgb(0.21633150526420653,0.35127534578175035,0.550589725728845)  );
  draw(  ellipsoid( (0.8955437178997936,0.3771299350778512,-0.23616617327295308), (0.27421244771084,-0.8857200284106134,-0.37457651393638164), (0.3504411260834604,-0.2706899394888499,0.8966147298638801), 0.5502743111335953, 0.8307228384030513, 0.9185039934712597,  (gDx*0, gDy*2, gDz*2)), rgb(0.21670407489617605,0.35047917224009884,0.5504754322323709)  );
  draw(  ellipsoid( (0.8733366195748886,0.4033590719356186,-0.27310182715745923), (0.3340827436313916,-0.90399156183987,-0.26681075040191793), (0.3545022839379573,-0.14177699111474001,0.9242442401623193), 0.5325628196122844, 0.787346757191086, 0.8639420299021159,  (gDx*0, gDy*2, gDz*3)), rgb(0.22212940004141368,0.3388588063158209,0.5487013080428987)  );
  draw(  ellipsoid( (-0.9456905643689568,-0.28983021944507154,0.1471998653530145), (-0.09675699839432071,0.6832754671536561,0.723721437604051), (-0.3103343998419209,0.6701739176380819,-0.674210264229476), 0.5026085325180423, 0.6781847651587908, 0.7945732321861795,  (gDx*0, gDy*3, gDz*0)), rgb(0.23144551945824993,0.3186084210362514,0.5449369638410367)  );
  draw(  ellipsoid( (-0.9282895000815078,-0.3066781009842381,0.2103025116709874), (-0.09632483059551133,0.7445535372990311,0.6605766852578485), (-0.35916588235684394,0.5929490471129888,-0.720701947047896), 0.5033648236448153, 0.6975929527566506, 0.7678100568948877,  (gDx*0, gDy*3, gDz*1)), rgb(0.235923621102639,0.3086289784496535,0.542731589353525)  );
  draw(  ellipsoid( (0.8927588081790492,0.3389630147230776,-0.29679249496671284), (-0.2234213012174415,0.9051335394435509,0.3616852194059017), (0.39123475380010486,-0.25658789999575726,0.8837974977305257), 0.4900408429624304, 0.6963811862421074, 0.7282051926652744,  (gDx*0, gDy*3, gDz*2)), rgb(0.2376951839334424,0.30462198028416054,0.5417772960124317)  );
  draw(  ellipsoid( (0.8269494232691256,0.39102825248159806,-0.40404400393527473), (0.4909207753690498,-0.8524565353971206,0.17976275356643898), (0.2741376363542807,0.3470083010821555,0.896902333208117), 0.4659087516467632, 0.6527365088347147, 0.7005493615324766,  (gDx*0, gDy*3, gDz*3)), rgb(0.23728212471472443,0.305560817550296,0.5420058717581394)  );
  draw(  ellipsoid( (-0.9403644307026734,-0.31564911485548613,0.12680841360172007), (-0.1988758045698463,0.8125803423078439,0.5478700590939092), (-0.27597672334081574,0.48997839093761597,-0.8268966226731438), 0.3780269233562201, 0.5543807153907138, 0.6603194623252717,  (gDx*1, gDy*0, gDz*0)), rgb(0.21182211459746333,0.36090749712888914,0.5518601856023564)  );
  draw(  ellipsoid( (0.942050285217889,0.3209826525726909,-0.09752639062475951), (-0.21709261274980074,0.8049313728300328,0.5522284694973847), (0.25575781045817364,-0.49905472824266345,0.8279687920497258), 0.38642480397931256, 0.5649264469488414, 0.6890293565098088,  (gDx*1, gDy*0, gDz*1)), rgb(0.2088810241439585,0.3671995812875356,0.5526106312108955)  );
  draw(  ellipsoid( (-0.9445880694861107,-0.3166627236892229,0.08647599904149937), (-0.2150949490444531,0.796090882283664,0.5656619750702548), (-0.24796681608648063,0.5157170024059136,-0.8200905020480315), 0.3941805469581515, 0.57372391932742, 0.706736429880581,  (gDx*1, gDy*0, gDz*2)), rgb(0.2082067194256194,0.3686452083454572,0.552773551694631)  );
  draw(  ellipsoid( (-0.9451613761111,-0.31603809897898605,0.08240080765089822), (-0.21691227540253127,0.7960349761835999,0.5650463533835715), (-0.24417010031424466,0.516186242248013,-0.82093405668568), 0.40042095198570543, 0.5798248652791497, 0.7115385438758675,  (gDx*1, gDy*0, gDz*3)), rgb(0.20971690968116422,0.3654099859219756,0.5524021044465607)  );
  draw(  ellipsoid( (-0.9353047221490854,-0.32308066104545785,0.14430510449756098), (-0.18535624642737974,0.7947423610154558,0.5779512449315526), (-0.3014102497195049,0.5138126760464455,-0.8032113017743315), 0.4205631282852376, 0.6378235213704248, 0.7520399543473495,  (gDx*1, gDy*1, gDz*0)), rgb(0.2069810589199741,0.37127509371536355,0.5530637187773816)  );
  draw(  ellipsoid( (0.9327427031793682,0.33734491023229274,-0.1272378135854726), (-0.21434097871912977,0.8026048785512242,0.5566716750181386), (0.2899120461577457,-0.4919591654458136,0.8209306822299235), 0.42798085340062064, 0.6469269310417565, 0.7727535126666045,  (gDx*1, gDy*1, gDz*1)), rgb(0.20579308859900813,0.3738303640945253,0.5533331743889975)  );
  draw(  ellipsoid( (0.9316982087753624,0.3427374656076656,-0.12028914096299621), (-0.22473616746669212,0.804091675590589,0.5503909812835798), (0.28536310693073813,-0.4857649708601407,0.8261962783068455), 0.43201771696166513, 0.6500211437615966, 0.7747738245762751,  (gDx*1, gDy*1, gDz*2)), rgb(0.20693138843823028,0.3713816710854041,0.5530754779380301)  );
  draw(  ellipsoid( (0.9322670338125362,0.3379507443651137,-0.12910256406997134), (-0.21399873222876137,0.8028868324767161,0.5563966901770612), (0.2916894243383851,-0.49108250693602523,0.8208259566501113), 0.43188930332372444, 0.6459055985167069, 0.7571716919358291,  (gDx*1, gDy*1, gDz*3)), rgb(0.21040485053037533,0.3639371338378933,0.5522304851602414)  );
  draw(  ellipsoid( (-0.9305395318079267,-0.32933773258857013,0.1601025847264936), (-0.1626633021273836,0.7634566848276789,0.6250396311698527), (-0.32808052347510686,0.5555812706394531,-0.7639977891531823), 0.4346697682304624, 0.6694995956777329, 0.7768863489409252,  (gDx*1, gDy*2, gDz*0)), rgb(0.206289784513288,0.3727613822779317,0.5532216657541673)  );
  draw(  ellipsoid( (-0.9244846302334689,-0.3459901234044167,0.16005937326093306), (-0.1854644683208393,0.7750254288346313,0.6041014117266772), (-0.33306320639265297,0.5287971436820492,-0.7806679712791803), 0.439337646509208, 0.6798447298952541, 0.785470910508058,  (gDx*1, gDy*2, gDz*1)), rgb(0.20600701548088207,0.373369953730298,0.5532851476308436)  );
  draw(  ellipsoid( (0.9189883832899662,0.35609201062567764,-0.16928919441788431), (-0.19792851989847218,0.7879792161735438,0.5830206307578899), (0.3410053553668831,-0.5022820271989527,0.7946245105482314), 0.4373309244284178, 0.6747283676769831, 0.7682268647068387,  (gDx*1, gDy*2, gDz*2)), rgb(0.208485722151478,0.3680465554692979,0.5527074996299125)  );
  draw(  ellipsoid( (0.912364402886223,0.36063201124805766,-0.19374144835126736), (-0.19804489823223612,0.8030092338935109,0.5620982018881934), (0.3582867770798257,-0.47446888490704653,0.8040583701602857), 0.42989344538658625, 0.6557922090640783, 0.7248064251942087,  (gDx*1, gDy*2, gDz*3)), rgb(0.21463596198965595,0.3548971188463625,0.5510855455919256)  );
  draw(  ellipsoid( (0.9328328252917202,0.3218965941071116,-0.1618811377539346), (-0.10970306057288823,0.6816918248307446,0.7233681596945436), (0.34320279509235974,-0.6570227078733054,0.6712175524966709), 0.4224220093440431, 0.6312803835606726, 0.7344248654742335,  (gDx*1, gDy*3, gDz*0)), rgb(0.21162746688196704,0.3613235411888611,0.5519112137561016)  );
  draw(  ellipsoid( (-0.9229796095435099,-0.3369154427770586,0.1860016795226555), (-0.10832045021177311,0.6911993834946712,0.7144998896588473), (-0.3692902929036031,0.6393210435101634,-0.6744577695395368), 0.42284720379698765, 0.6427464271159259, 0.7189203090505242,  (gDx*1, gDy*3, gDz*1)), rgb(0.21375197924594105,0.3567850684942173,0.5513349232352259)  );
  draw(  ellipsoid( (-0.9078286894696713,-0.3483810943948074,0.23340454932167745), (-0.08565888505680433,0.6989241017836667,0.7100476430189174), (-0.41049923993425896,0.6246084477608121,-0.6643452874817618), 0.4165707788973127, 0.637273243258973, 0.6842443647493123,  (gDx*1, gDy*3, gDz*2)), rgb(0.2178394622247308,0.34805287317288,0.5501271285154342)  );
  draw(  ellipsoid( (-0.8787974718046102,-0.36526694857097447,0.3070750068492623), (-0.13502434700017452,0.8075378987059836,0.5741523908081384), (-0.4576935975813971,0.46310106719286936,-0.7589821949807674), 0.403724493343817, 0.6210640421629454, 0.6349807426053411,  (gDx*1, gDy*3, gDz*3)), rgb(0.22267603211698944,0.33768224853745094,0.5485039444990829)  );
  draw(  ellipsoid( (-0.9645129842628704,-0.2640311222476916,0.0015065433803720296), (-0.17304563461204187,0.6364294801020282,0.751673283547299), (-0.19942394923892212,0.7247379411495488,-0.6595339302327554), 0.24306414802407958, 0.3604712045545993, 0.50254801543312,  (gDx*2, gDy*0, gDz*0)), rgb(0.18791046452520083,0.413205559149722,0.5564582519739543)  );
  draw(  ellipsoid( (-0.9618396952674736,-0.273057048240474,-0.017442735277841594), (-0.19694828160411013,0.6466747771430322,0.7369010157274841), (-0.18993623925139028,0.7122159651490614,-0.6757755870152676), 0.2473353905294337, 0.3669667268371991, 0.531359559413343,  (gDx*2, gDy*0, gDz*1)), rgb(0.18307241377343553,0.42429411647208987,0.5570324915809228)  );
  draw(  ellipsoid( (-0.9599891246619259,-0.2790053401817809,-0.02401875685538772), (-0.21040994054707374,0.6620447861739268,0.7193221517643512), (0.18479322890681504,-0.6955952280241494,0.6942612918060643), 0.2534501125984139, 0.373793726613748, 0.5520133803600114,  (gDx*2, gDy*0, gDz*2)), rgb(0.18139693682517175,0.4281856647177468,0.557205536714396)  );
  draw(  ellipsoid( (-0.9582064831096022,-0.2846943852243719,-0.02809702383607625), (-0.22285507007081143,0.6812536229465127,0.6973013114615407), (0.1793765689003505,-0.6744202015390461,0.7162272253171966), 0.2610690093733771, 0.3792272907364219, 0.5630954718836345,  (gDx*2, gDy*0, gDz*3)), rgb(0.1825811957732798,0.4254312173360314,0.5570851434493926)  );
  draw(  ellipsoid( (-0.9615419466289018,-0.27212770439704764,0.03719673873735929), (-0.1508082044516879,0.6362836079724801,0.7565712495829995), (-0.22955167249044398,0.721865418707058,-0.6528524694995983), 0.28932560350776104, 0.4243727011895236, 0.5748631594271622,  (gDx*2, gDy*1, gDz*0)), rgb(0.19329100489608725,0.40112200460284897,0.5556907817975676)  );
  draw(  ellipsoid( (-0.9603695340981282,-0.2778222641699304,0.022475486816477683), (-0.16819579599294662,0.6419344431252824,0.7480844504063248), (-0.22226230491570853,0.7146572327073494,-0.6632228189323932), 0.29701080263726054, 0.43434353786701785, 0.5984797902231365,  (gDx*2, gDy*1, gDz*1)), rgb(0.19141328285934342,0.4053106081096267,0.5559743316550325)  );
  draw(  ellipsoid( (0.9605848755402684,0.2769348237806685,-0.024161958961493934), (-0.16798324324438899,0.6475233035457993,0.7433002094404597), (0.2214911440044123,-0.7099441349449299,0.6685215018119552), 0.30317757492573616, 0.44094770914821446, 0.6096420196413448,  (gDx*2, gDy*1, gDz*2)), rgb(0.19173358351076986,0.4045939214840132,0.5559273814436705)  );
  draw(  ellipsoid( (-0.9603099289974708,-0.2769304731121497,0.03338193120168309), (-0.16568023691114547,0.6625773048558191,0.730439165288298), (-0.22439897362942562,0.6959172566849312,-0.682161470974563), 0.30794166004454826, 0.44377173382694923, 0.6050364034484798,  (gDx*2, gDy*1, gDz*3)), rgb(0.19501299394744598,0.39730507587751995,0.5554150822438568)  );
  draw(  ellipsoid( (-0.9564705088135272,-0.2797591351282094,0.08306017145605134), (-0.1161645744681572,0.626081601831768,0.7710561714209928), (-0.26771245284563117,0.7278438391304027,-0.6313266891485675), 0.32844840327391756, 0.48271305785123947, 0.6295873676259722,  (gDx*2, gDy*2, gDz*0)), rgb(0.19824180746286174,0.39020393586294266,0.554858576900582)  );
  draw(  ellipsoid( (-0.9542564254833438,-0.2901714720240693,0.07207767509524857), (-0.13514374225657633,0.6336441301796407,0.761729141636169), (-0.26670366205329726,0.7171440811387012,-0.6438738413191581), 0.33572796721046433, 0.4911569645159657, 0.641279247017928,  (gDx*2, gDy*2, gDz*1)), rgb(0.19882321532190853,0.3889331999569658,0.5547526012205886)  );
  draw(  ellipsoid( (-0.9539305949698228,-0.2898513794189275,0.07747643402651383), (-0.1285197424136814,0.6281063050399562,0.7674406461603231), (-0.27110716661578793,0.7221278608515248,-0.6364214458922328), 0.3386137616990191, 0.49360798333184136, 0.6374262286932554,  (gDx*2, gDy*2, gDz*2)), rgb(0.20098330084898677,0.3842264884231835,0.5543431892230805)  );
  draw(  ellipsoid( (-0.9540197823297917,-0.2815240021521581,0.10291011192127036), (-0.1006590445649695,0.624291506834623,0.7746792053756338), (-0.2823366991212516,0.7287004533458087,-0.6239243845394478), 0.33928147814272525, 0.4915905384223478, 0.6189227967371502,  (gDx*2, gDy*2, gDz*3)), rgb(0.20574672251797552,0.37393015250853406,0.5533435836117228)  );
  draw(  ellipsoid( (0.9516788912812596,0.2818819279055979,-0.12186002876208282), (-0.0751355376666469,0.5984815433848782,0.7976054746597455), (0.29776154699560553,-0.7499082750229372,0.590741601872516), 0.35010912614624645, 0.5106411565794147, 0.6470991733999172,  (gDx*2, gDy*3, gDz*0)), rgb(0.20364240938028672,0.37846586128275106,0.5538015606102867)  );
  draw(  ellipsoid( (0.9480059813799643,0.28874047714054707,-0.13384168307541508), (-0.07444199577029015,0.6100689909595223,0.7888435938355374), (0.3094237362154797,-0.7378650033239684,0.5998434698620746), 0.3564497052346803, 0.519826889843339, 0.6447330418434243,  (gDx*2, gDy*3, gDz*1)), rgb(0.20683453356494813,0.3715894914508933,0.553098407693783)  );
  draw(  ellipsoid( (-0.9440009344481349,-0.2935591291462001,0.15061631205142612), (-0.057661443905909486,0.5962503111278107,0.8007251241010706), (-0.3248651930381794,0.7472005113588411,-0.5797878941279756), 0.3565628884336757, 0.5188337748759742, 0.627385440906383,  (gDx*2, gDy*3, gDz*2)), rgb(0.21104005223016334,0.36257909343616257,0.552065208294841)  );
  draw(  ellipsoid( (-0.9366070827460743,-0.28852200932343597,0.19880196851604753), (0.001516563800608902,0.5640436916545346,0.8257435521631173), (-0.3503781850859539,0.7736987553568192,-0.5278497545468614), 0.35202594432955836, 0.5100637958969577, 0.5965638477860258,  (gDx*2, gDy*3, gDz*3)), rgb(0.21691800541670492,0.35002200717461446,0.5504098045883129)  );
  draw(  ellipsoid( (-0.9697990507661309,-0.1914746757414486,-0.1510868944773789), (-0.21456800285512434,0.3752094857238225,0.9017640567097515), (0.11597574437732648,-0.9069483394177265,0.4049621418645866), 0.17185453355294758, 0.25125247012478275, 0.4321379002081306,  (gDx*3, gDy*0, gDz*0)), rgb(0.1651809525398031,0.4672646108764209,0.5581400620294162)  );
  draw(  ellipsoid( (0.9618722295218523,0.21743147462213666,0.16590770903820282), (-0.24730248948611006,0.4323744787153334,0.8671180939466181), (-0.11680450662810125,0.8750862037532572,-0.46966034879695656), 0.17306470684972075, 0.25396005568125773, 0.45092506807164323,  (gDx*3, gDy*0, gDz*1)), rgb(0.16161121680764828,0.4761683719553731,0.5581310158565953)  );
  draw(  ellipsoid( (0.9563990605148011,0.23435506179976312,0.17429441200232224), (-0.2685895101054205,0.4713424780469512,0.8400571072551493), (-0.11471927522319991,0.8502434788914225,-0.5137368144243463), 0.17659188572295728, 0.2589350565492195, 0.4727315976469295,  (gDx*3, gDy*0, gDz*2)), rgb(0.1589349602684682,0.4828903459440341,0.5580623908642377)  );
  draw(  ellipsoid( (0.9541799340273334,0.24294412835263476,0.17469631936183466), (-0.2777464222459775,0.5018235536722151,0.8191642362245489), (-0.11134441355161237,0.8301513545605813,-0.5463068277919283), 0.18151750985827986, 0.263181102964035, 0.49230964754789375,  (gDx*3, gDy*0, gDz*3)), rgb(0.15750867997611356,0.48648951989594835,0.5580013724182047)  );
  draw(  ellipsoid( (-0.9789978816403221,-0.17941360414671337,-0.09681893611710628), (-0.16349678823066763,0.40724161622617383,0.8985672296783086), (0.12178648521908363,-0.895524999461534,0.42802222764408104), 0.2077705419362991, 0.28678427352642966, 0.47349134431572537,  (gDx*3, gDy*1, gDz*0)), rgb(0.17533797509387633,0.4424950768969932,0.55771934822269)  );
  draw(  ellipsoid( (-0.9740791279550732,-0.196984611401645,-0.11120663358463433), (-0.19039387275193181,0.44848460038104193,0.8732764375841011), (0.12214755708079143,-0.8718134124295178,0.474364151477738), 0.21411489235910605, 0.2928792640494239, 0.4973530008409738,  (gDx*3, gDy*1, gDz*1)), rgb(0.1729237013416641,0.4482977421689933,0.5578727574117204)  );
  draw(  ellipsoid( (-0.9700575641216584,-0.21138290857102318,-0.11960597081420722), (-0.20964348692809678,0.48011129999975666,0.8517879712705344), (0.12262924071763458,-0.8513579773377343,0.5100506482148789), 0.21872521523158997, 0.29679958639979676, 0.5134408785156777,  (gDx*3, gDy*1, gDz*2)), rgb(0.17154613689112347,0.4516330837097144,0.5579458094936554)  );
  draw(  ellipsoid( (-0.9678141385804406,-0.21980797725861553,-0.12255711442946146), (-0.22024935120740072,0.5041627699113673,0.8350509713352933), (0.12176213063395094,-0.835167261432567,0.5363484212475197), 0.2237504584806407, 0.30066532336899904, 0.5221037750991535,  (gDx*3, gDy*1, gDz*3)), rgb(0.17203897492836512,0.45043883781130445,0.5579202572947056)  );
  draw(  ellipsoid( (-0.9841060362758868,-0.17697861336313914,-0.014624629137999863), (-0.0938436557198754,0.44837150384127006,0.8889073983403714), (0.1507603318067886,-0.8761515650589106,0.457853423486522), 0.25392021539261594, 0.33789942016768865, 0.5244643701023064,  (gDx*3, gDy*2, gDz*0)), rgb(0.18744054216478434,0.41427093868551257,0.5565196307219755)  );
  draw(  ellipsoid( (-0.9829183415757565,-0.1825810460308004,-0.02314509503681571), (-0.10959088006364719,0.4796147614372548,0.8706087063763837), (0.14785591905919707,-0.8582737571671312,0.49143136342461546), 0.2627905608749487, 0.3463429765992964, 0.54742110217468,  (gDx*3, gDy*2, gDz*1)), rgb(0.18606105714582338,0.4174169620742516,0.5566908855092301)  );
  draw(  ellipsoid( (-0.9814711903107428,-0.18979400482942446,-0.026316122830350074), (-0.11956382546562633,0.499307965266787,0.8581352151387137), (0.14972906942220277,-0.8453814473667787,0.5127488802698759), 0.26869648318267325, 0.3495660815335609, 0.5582932161634215,  (gDx*3, gDy*2, gDz*2)), rgb(0.1861257177590342,0.41726934285937506,0.556682933218889)  );
  draw(  ellipsoid( (-0.9802290864849981,-0.19657725814912808,-0.02254594392251803), (-0.12292867549106333,0.5157358945108039,0.8478826734019005), (0.15504669861302786,-0.8338908014192935,0.5297043067197862), 0.2718471127471179, 0.3505646630716449, 0.5559352716029596,  (gDx*3, gDy*2, gDz*3)), rgb(0.18816816126146008,0.41262132470510593,0.5564245930030836)  );
  draw(  ellipsoid( (-0.9824680860041175,-0.17427639737488132,0.06621327133920173), (-0.029920429796071703,0.4979560174247041,0.8666859711518042), (0.18401420560441673,-0.84951017770746,0.4944403200662497), 0.29866811120928566, 0.39265524776528227, 0.5702141319662741,  (gDx*3, gDy*3, gDz*0)), rgb(0.19901509809839144,0.3885138173668936,0.5547176259318986)  );
  draw(  ellipsoid( (-0.9821154291330626,-0.17619059500692336,0.06637889792612345), (-0.03677892585643879,0.5252925640873896,0.8501264804294374), (0.1846526319109127,-0.83248098857908,0.5223780328294543), 0.30737981454322394, 0.40109689774187196, 0.5846557510246464,  (gDx*3, gDy*3, gDz*1)), rgb(0.19948986755984421,0.3874767078037,0.554630483189151)  );
  draw(  ellipsoid( (-0.9819476044609924,-0.1726382825072514,0.0772976423072848), (-0.027458941985306885,0.5344238170473865,0.8447704956244324), (0.187149428528024,-0.8273978530220202,0.5295166515003901), 0.31235929727687983, 0.4033778988224832, 0.5893520465135786,  (gDx*3, gDy*3, gDz*2)), rgb(0.20060821856818506,0.38504279518908535,0.5544153443550424)  );
  draw(  ellipsoid( (-0.9806457933199744,-0.16576326138508682,0.10419390202318676), (-0.001935757140929917,0.5403535934973335,0.8414358245515873), (0.19578069590265895,-0.8249488076053557,0.5302163557854213), 0.31353474108317414, 0.4013165493779555, 0.5783072587075675,  (gDx*3, gDy*3, gDz*3)), rgb(0.20421889753881095,0.37722202541897853,0.5536787435678185)  );
