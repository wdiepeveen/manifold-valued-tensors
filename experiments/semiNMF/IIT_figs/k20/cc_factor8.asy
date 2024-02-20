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

  draw(  ellipsoid( (-0.4598114142028221,-0.6361051677094575,-0.6196318899008704), (0.7567149216522447,-0.6458242801705029,0.10145701794892803), (-0.46471065268362205,-0.42223360211567174,0.7783076477375683), 0.10263913601765079, 0.23035868441719984, 0.6608748147187914,  (gDx*0, gDy*0, gDz*0)), rgb(0.12149060826869036,0.6299264816219694,0.5317860815201854)  );
  draw(  ellipsoid( (0.4816370986977647,0.5623883794858794,0.6721197927283843), (-0.7301255171161894,0.6816819181135769,-0.04718571577089331), (-0.48470860773645896,-0.46800541998596706,0.7389374076672138), 0.17220415278876, 0.2916513478356675, 0.7391817286262208,  (gDx*0, gDy*0, gDz*1)), rgb(0.12519721787474514,0.5750436710204594,0.548928613695617)  );
  draw(  ellipsoid( (0.5065862526993339,0.48935950381523846,0.7098574818946055), (-0.711810114784792,0.7019595731738314,0.024064873148923215), (-0.4865148806214545,-0.5174746695784405,0.7039341143023086), 0.28151574830710113, 0.395615367347308, 0.8382528711126955,  (gDx*0, gDy*0, gDz*2)), rgb(0.14831454311210468,0.5099137445958315,0.5571483129647578)  );
  draw(  ellipsoid( (0.553803961843376,0.4081933761428866,0.7257267664346221), (-0.6923264696888585,0.7099617234259696,0.12898996332346524), (-0.4625853772167282,-0.5738750028909592,0.6757826942458344), 0.431079303382915, 0.5405172968604873, 0.9312568124671049,  (gDx*0, gDy*0, gDz*3)), rgb(0.1797820852049823,0.43196393468320626,0.5573598530370576)  );
  draw(  ellipsoid( (-0.4280252553394878,-0.62204012756836,-0.6556374459152747), (0.7712876320787148,-0.6295434700876963,0.093757175044638), (-0.47107299794615204,-0.4655546143737386,0.7492323615817353), 0.058270791452738266, 0.19901586055496306, 0.6230410816868208,  (gDx*0, gDy*1, gDz*0)), rgb(0.1566799353402579,0.6827108346429109,0.50242172573771)  );
  draw(  ellipsoid( (0.4432958342464055,0.5403590772985086,0.7151928907091342), (-0.7301922032163077,0.6804657251993486,-0.061528393372806564), (-0.5199116749043988,-0.4949529921215036,0.6962136065088598), 0.0753555131649406, 0.20605947215399295, 0.6354214810698152,  (gDx*0, gDy*1, gDz*1)), rgb(0.13559617013621234,0.6598711640394952,0.5169391905768609)  );
  draw(  ellipsoid( (0.43355517096846385,0.4703589121412246,0.7686302150552173), (0.7140798088863821,-0.6996057415604835,0.025334421572122928), (-0.5496543825598901,-0.5378794475948393,0.6391914889825868), 0.11939138622095989, 0.24134286933828342, 0.6784053624178726,  (gDx*0, gDy*1, gDz*2)), rgb(0.1194799215319676,0.614628392525215,0.5377581870626966)  );
  draw(  ellipsoid( (0.4187068936456207,0.4071844187103256,0.811717553323303), (-0.7170795297168229,0.6966925790493169,0.020405841288986343), (-0.5572086550608623,-0.590610107818301,0.5836935970765139), 0.20622901783363703, 0.31402242769966615, 0.7416056994221073,  (gDx*0, gDy*1, gDz*3)), rgb(0.13458554941039594,0.5461901270377341,0.5538517816419298)  );
  draw(  ellipsoid( (0.37072781427224905,0.6197724341040787,0.6916957551189877), (0.7995955924317758,-0.5918554313714263,0.10175478819112388), (-0.4724487022826907,-0.5153535468790806,0.7149845770578223), 0.05836823502256708, 0.21788827630844765, 0.6325547115190041,  (gDx*0, gDy*2, gDz*0)), rgb(0.15843163432053803,0.6842600378084587,0.5013295337436696)  );
  draw(  ellipsoid( (0.37377926946237233,0.5505530785073358,0.7464451523496354), (0.7614489118741576,-0.6416660980950629,0.09197919961120846), (-0.5296079798547567,-0.5339999309939581,0.6590595279431224), 0.05303924791515767, 0.20316049730251845, 0.6031517902505461,  (gDx*0, gDy*2, gDz*1)), rgb(0.16344123017174284,0.6884713274119464,0.4982716426239596)  );
  draw(  ellipsoid( (0.36061979026238017,0.4708365350017679,0.8051498768419739), (-0.7301523257394396,0.6796464131377783,-0.07041544096460606), (-0.580371388068368,-0.5624888535931015,0.5888763380329646), 0.05915251126064196, 0.19213859134007621, 0.5913928072815904,  (gDx*0, gDy*2, gDz*2)), rgb(0.15005645166782347,0.6765381816031926,0.5066498335038556)  );
  draw(  ellipsoid( (0.3392020530720243,0.38305917565454906,0.8591901041903248), (-0.7138323930048159,0.6996684542650876,-0.03012256302709354), (-0.6126869362804647,-0.603100092897655,0.5107690241763861), 0.08802801857380596, 0.20626947738102824, 0.6132369384627427,  (gDx*0, gDy*2, gDz*3)), rgb(0.12434805400566618,0.6392802613987222,0.5276160153146675)  );
  draw(  ellipsoid( (0.3432300264295287,0.5782042987694234,0.7401843944867686), (-0.8021778375307412,0.5903834084686995,-0.08920845239877406), (-0.48857329637597774,-0.5631404974688287,0.6664599869301018), 0.07517729832372418, 0.24137569546154394, 0.653556016845036,  (gDx*0, gDy*3, gDz*0)), rgb(0.13802691695244695,0.6631162690517158,0.5150572786673472)  );
  draw(  ellipsoid( (0.317580510394299,0.5371541352086647,0.7814141376030562), (-0.7836407319665273,0.6126708596043519,-0.1026723964589802), (-0.5339005737219824,-0.5797411946894029,0.615508183990629), 0.06268309273131141, 0.23866411175429425, 0.623186975081722,  (gDx*0, gDy*3, gDz*1)), rgb(0.15048885473295515,0.6769560221852952,0.5063690731741717)  );
  draw(  ellipsoid( (0.30072711704199545,0.4678449383034542,0.8310741933061374), (0.7503987529389753,-0.6538931788619118,0.09656822575513524), (-0.5886127017515344,-0.5945963541274879,0.5477136688042924), 0.056249459975041945, 0.22176401752787828, 0.5883300775606891,  (gDx*0, gDy*3, gDz*2)), rgb(0.15553678142471608,0.68168179389306,0.503139916446337)  );
  draw(  ellipsoid( (0.2755952945617228,0.3843954696990203,0.8810717090511456), (0.7227560164037141,-0.6871601148593058,0.07372053512271631), (-0.6337751765158298,-0.6164828460074253,0.4672022326690163), 0.05888693991601593, 0.20344409175290235, 0.5734163685294643,  (gDx*0, gDy*3, gDz*3)), rgb(0.14787158371063752,0.6743230026239505,0.5081016696011138)  );
  draw(  ellipsoid( (0.4939561616475763,0.5379306869367934,0.6831089857571067), (-0.7593386565038989,0.6496127386659469,0.03752458529793578), (-0.42357067309865476,-0.5372465596117276,0.7293517800596054), 0.303727482325192, 0.3834103980870149, 0.6508413936804698,  (gDx*1, gDy*0, gDz*0)), rgb(0.18107049406758752,0.4289462944005997,0.5572380405415187)  );
  draw(  ellipsoid( (0.6046062397210841,0.402380497978954,0.6874163438095868), (-0.6817534328416768,0.7077085203767426,0.18536695227238084), (-0.41190235699607103,-0.5807224681656216,0.7022092731292501), 0.4406558561761702, 0.5146894678160105, 0.7696295563779247,  (gDx*1, gDy*0, gDz*1)), rgb(0.2109486950072662,0.36277436225719634,0.5520891581799132)  );
  draw(  ellipsoid( (0.7047994675371894,0.3058070609039626,0.6401091719859777), (0.618091640346354,-0.7075579266276156,-0.34252664801498717), (-0.3481672510349687,-0.6370587272518204,0.6877032378424893), 0.5926696262579156, 0.6721915608358695, 0.8911165289237135,  (gDx*1, gDy*0, gDz*2)), rgb(0.24097252075989015,0.29709619739782134,0.5398677434889079)  );
  draw(  ellipsoid( (-0.8127781337487462,-0.2734019973090839,-0.5144346928107677), (-0.5393103027693096,0.6870455940163698,0.48694224407990033), (-0.22030910699918985,-0.6732159383215,0.7058641496514195), 0.7440485488018648, 0.8400967900950993, 0.9999999999999994,  (gDx*1, gDy*0, gDz*3)), rgb(0.2647227565332179,0.2342879758483174,0.5172152703229755)  );
  draw(  ellipsoid( (0.4477832679205635,0.5579098341631267,0.6987322533806877), (-0.7783838892984009,0.62778388386147,-0.0024322920266774033), (-0.4400098474478449,-0.5427927892924524,0.7153791456570728), 0.14077420510965308, 0.26523454193240653, 0.5784900857585011,  (gDx*1, gDy*1, gDz*0)), rgb(0.1280661577735234,0.5653682823103703,0.5508447684595876)  );
  draw(  ellipsoid( (0.4764650604009721,0.4788188067565705,0.7373693758987498), (-0.7340421104737904,0.6782582671490107,0.03388071863471764), (-0.48390414987710323,-0.5574031515341787,0.6746395336707472), 0.21765060438536601, 0.32519917416102895, 0.6318763265287353,  (gDx*1, gDy*1, gDz*1)), rgb(0.1516082897302677,0.5014762431579014,0.5575532798111318)  );
  draw(  ellipsoid( (0.5133181285840487,0.40178498487577036,0.7583358918021401), (-0.704791275174433,0.7015445016543248,0.10537822657732289), (-0.4896669861293157,-0.5885610742539485,0.6432900625441614), 0.33967985822675795, 0.43099987894276454, 0.7197948526595362,  (gDx*1, gDy*1, gDz*2)), rgb(0.1827347601471146,0.4250757373208388,0.5570686834434274)  );
  draw(  ellipsoid( (0.5619412828058551,0.3443281670112081,0.7521037881045486), (-0.6984953754370411,0.684574492042872,0.20847535906507286), (-0.44308713045131154,-0.6424919285461455,0.6252023005247798), 0.49994045441651863, 0.582314288584563, 0.8228113388038237,  (gDx*1, gDy*1, gDz*3)), rgb(0.22268280628815545,0.33766766797378933,0.5485014986593901)  );
  draw(  ellipsoid( (0.40044048544962285,0.5714047784427503,0.7163406988198155), (-0.8042839117769064,0.5937594231694283,-0.024023668627943053), (-0.4390612791737713,-0.5665212498841495,0.6973369820682054), 0.0745382847152062, 0.22191981369476105, 0.5613556044927337,  (gDx*1, gDy*2, gDz*0)), rgb(0.12814313931800575,0.6478520638489651,0.5234376109415128)  );
  draw(  ellipsoid( (0.4145964715790408,0.493010361390904,0.7648859714463456), (-0.7517760665885309,0.659189080452721,-0.017392582228436926), (-0.5127792034188657,-0.5678120637791229,0.6439308571329743), 0.09051107445981263, 0.2255210841756032, 0.5501505966602278,  (gDx*1, gDy*2, gDz*1)), rgb(0.12006921772631964,0.6220477729668047,0.5349892119513775)  );
  draw(  ellipsoid( (0.41910425659225686,0.4119416606867618,0.8091079596054438), (-0.7090702993375275,0.7050888972413886,0.008304070346020691), (-0.5670722464586327,-0.5771946943428671,0.5876013547625669), 0.13541192649703138, 0.25388783814913674, 0.5716482891489432,  (gDx*1, gDy*2, gDz*2)), rgb(0.12664026788324034,0.5700142601956435,0.5499610865798358)  );
  draw(  ellipsoid( (0.40825731417886074,0.34908871358751636,0.8434826823742594), (-0.7005021604820544,0.7122761028859125,0.044265973587296524), (-0.5853398060780053,-0.6089333488186291,0.5353293267859448), 0.23111078213118894, 0.3322635178506551, 0.6300958674282134,  (gDx*1, gDy*2, gDz*3)), rgb(0.1568743919854812,0.48809458587361443,0.5579678973152036)  );
  draw(  ellipsoid( (0.35244056543069613,0.5613998466848557,0.7487428530417581), (-0.8286336732647452,0.5590337483401498,-0.02911191763758067), (-0.4349159497772721,-0.6101733199337567,0.6622209874885807), 0.06594814635171212, 0.23553221429576246, 0.594894350621195,  (gDx*1, gDy*3, gDz*0)), rgb(0.14162381851391245,0.6675045695603248,0.5124158889257114)  );
  draw(  ellipsoid( (0.3550613247379616,0.5004287011038467,0.7896217897112773), (-0.7856175479123568,0.617537708819436,-0.0381083533915755), (-0.5066917446411577,-0.6068099317723449,0.6124093260352441), 0.0610179997095892, 0.22450547947392194, 0.5549304012570568,  (gDx*1, gDy*3, gDz*1)), rgb(0.14255554231449602,0.6685890201526627,0.5117495542839751)  );
  draw(  ellipsoid( (0.3539230339792215,0.42127395701184867,0.8350249931364461), (-0.7325916882079129,0.6798919476543585,-0.03250165970290314), (-0.5814188717161002,-0.6002292834062183,0.5492512202572106), 0.06629606114087626, 0.2105711849563285, 0.5277277397869219,  (gDx*1, gDy*3, gDz*2)), rgb(0.13174625802802198,0.6541535168749295,0.5201225318579924)  );
  draw(  ellipsoid( (0.3285317051295358,0.3516660093469294,0.8765831030739101), (-0.6994858429125073,0.7142306204739237,-0.024375732653869938), (-0.63465461023273,-0.6051492697239578,0.4806327985717392), 0.09204189567459391, 0.22062824748699522, 0.5300807314301502,  (gDx*1, gDy*3, gDz*3)), rgb(0.11949565889752624,0.6150322598639529,0.5376131749204963)  );
  draw(  ellipsoid( (0.9634478716658287,-0.21878343016032295,0.1546027466439968), (-0.027611355865384893,0.4929309238074666,0.8696302187605905), (0.26646915695791146,0.8421121748565332,-0.46887234227296815), 0.4933689019282208, 0.534604480808458, 0.6257527303987559,  (gDx*2, gDy*0, gDz*0)), rgb(0.27386076238263635,0.20086402435448142,0.49958028494188217)  );
  draw(  ellipsoid( (0.9883407657239016,-0.15122539089670153,0.017703444761731447), (-0.030987855375669948,-0.3136236957792354,-0.949041585108361), (-0.14907140445943484,-0.9374278951438949,0.31465323417784546), 0.6146305240064327, 0.6915739679972616, 0.7436598662693236,  (gDx*2, gDy*0, gDz*1)), rgb(0.27989355120205817,0.1682969174782712,0.4783551356005852)  );
  draw(  ellipsoid( (-0.9981059148357346,-0.018206883995538145,0.05876301681409741), (0.035099051919979946,0.6159432467983316,0.7870082421918134), (-0.05052365113725914,0.7875801077342502,-0.6141375534658886), 0.7307174145504489, 0.8517713234258756, 0.8605082848885099,  (gDx*2, gDy*0, gDz*2)), rgb(0.280719515982417,0.16196323219338232,0.47377076831856624)  );
  draw(  ellipsoid( (-0.9690524532391231,-0.23889773629557817,0.0621708489895823), (-0.2335436528287442,0.9688312493707223,0.0826037073391548), (0.07996689999396163,-0.06552771807116141,0.9946413489644104), 0.8205963456967288, 0.9526515411604225, 0.9984314737988659,  (gDx*2, gDy*0, gDz*3)), rgb(0.27861143252422754,0.1767755596797817,0.4842668681450235)  );
  draw(  ellipsoid( (0.529987311869608,0.42609079856852844,0.7331848884370575), (-0.7733580575284281,0.5975593962913806,0.21175477033525605), (-0.3478947601115671,-0.6792417826364281,0.646219650434589), 0.3153569440792683, 0.36831782462842066, 0.559514796443471,  (gDx*2, gDy*1, gDz*0)), rgb(0.20797722162347532,0.3691376390874975,0.5528278837934782)  );
  draw(  ellipsoid( (-0.7465687822357372,-0.17282124252192851,-0.6424701327876845), (-0.5671740099504173,0.6701110975770774,0.4788159973735304), (-0.3477767902373763,-0.7218614375607146,0.5983037432065478), 0.4392685592600855, 0.47211682968307905, 0.6319415759041702,  (gDx*2, gDy*1, gDz*1)), rgb(0.24757566976059509,0.28136023626414997,0.5353584545553223)  );
  draw(  ellipsoid( (0.9444452158256149,-0.07893008666689738,0.3190505848966022), (-0.19745520763604787,0.6397647879490559,0.7427734897516907), (0.2627445057113722,0.7645070683252412,-0.5886376365806317), 0.5778554642093712, 0.6205801010188156, 0.7280504574254594,  (gDx*2, gDy*1, gDz*2)), rgb(0.27460702131202364,0.19755075857130225,0.49760146478575673)  );
  draw(  ellipsoid( (-0.9894653771377717,-0.04826453749717657,-0.13648736888734658), (-0.12405023159277043,0.768687690698761,0.627479700237375), (-0.07463114287874037,-0.6378007279616346,0.7665770828326541), 0.709620394812164, 0.7858123350376132, 0.8390249364367239,  (gDx*2, gDy*1, gDz*3)), rgb(0.28179427063844753,0.1518479232163568,0.4661460540135858)  );
  draw(  ellipsoid( (0.41534668260675733,0.5226397343069151,0.7445366622075494), (-0.827252581267215,0.5574278851509726,0.07019486906351384), (-0.37833886931465494,-0.6450750816608183,0.663880892168164), 0.1582727906393456, 0.27119920614364706, 0.525488485134884,  (gDx*2, gDy*2, gDz*0)), rgb(0.14170636085041285,0.5270542604941608,0.5559367411024568)  );
  draw(  ellipsoid( (0.4453238921910822,0.43742070806013494,0.7812488433296753), (-0.7724276008194928,0.6289565668158211,0.08814328421098129), (-0.4528158925391512,-0.6427104800914836,0.6179652144292157), 0.21968856983548704, 0.3075044622424967, 0.5308705761909932,  (gDx*2, gDy*2, gDz*1)), rgb(0.16883614270902908,0.4582472159645774,0.5580577056441648)  );
  draw(  ellipsoid( (0.494115217699047,0.340126079155223,0.8001027446001662), (-0.7172498408260733,0.6795720416173706,0.15406007298119445), (-0.49132760707413536,-0.6499969927088076,0.579742263420833), 0.3236696640762421, 0.38283942220110057, 0.5728923407108739,  (gDx*2, gDy*2, gDz*2)), rgb(0.20902160945531262,0.3668985941182802,0.5525755596624778)  );
  draw(  ellipsoid( (0.5987968483808379,0.24936202284937478,0.7610919234426572), (-0.6528628218802023,0.7024197805321184,0.28350765020335866), (-0.4639099806813769,-0.6666521082826324,0.5834059447301904), 0.4708074072402462, 0.5078714056721926, 0.6506668723976339,  (gDx*2, gDy*2, gDz*3)), rgb(0.25691736568556073,0.25723339458552824,0.5269850068073669)  );
  draw(  ellipsoid( (0.363919308586933,0.5339398114522567,0.7631978869100315), (0.853844597181503,-0.5186396500371495,-0.044298050445816216), (-0.3721721922733691,-0.6677733082084606,0.6446446060759081), 0.09617707526220134, 0.2497943405591093, 0.5481987390679747,  (gDx*2, gDy*3, gDz*0)), rgb(0.11947000839380896,0.6140210475940284,0.5379713195331075)  );
  draw(  ellipsoid( (0.3730273402518208,0.4676231692836648,0.8013608269523493), (-0.8029888655945208,0.5954116576549199,0.026340836354322706), (-0.4648219929778252,-0.6533096734916154,0.597601025238747), 0.10470580134200563, 0.24656157403091036, 0.5110063336120897,  (gDx*2, gDy*3, gDz*1)), rgb(0.1212585916353647,0.5921424852347239,0.5448000058359124)  );
  draw(  ellipsoid( (0.37837096298545436,0.3925734909125494,0.8382848373925145), (-0.7468172716834056,0.6645253446134366,0.02588491996795802), (-0.5468997870586667,-0.6358396972501223,0.5446177579881671), 0.13985145587021322, 0.25374350118502215, 0.4923099204786288,  (gDx*2, gDy*3, gDz*2)), rgb(0.13762941812060564,0.5378719426564249,0.5548623391079414)  );
  draw(  ellipsoid( (0.3616009015303459,0.3408388182057259,0.8677981839209882), (-0.7139774752935585,0.699802235072304,0.022649427393282753), (-0.5995673046338306,-0.6277784097847996,0.4964003579996608), 0.21254242400758558, 0.300177279351449, 0.5100756350070409,  (gDx*2, gDy*3, gDz*3)), rgb(0.16973507378679442,0.45604472982201516,0.5580262158195153)  );
  draw(  ellipsoid( (-0.9477089417978938,-0.013440198985034048,0.3188528229254977), (0.22249738079615639,-0.7440796895113315,0.629952642026026), (0.22878522062223877,0.6679556697176571,0.708161384231638), 0.4974967093885125, 0.5801737512108208, 0.6494098576332161,  (gDx*3, gDy*0, gDz*0)), rgb(0.26986857284760035,0.2167494992593207,0.5084771970161686)  );
  draw(  ellipsoid( (0.9452318933947595,0.11881348672131739,-0.30400661683996544), (-0.24246244330465336,0.8791334108408116,-0.410288203011174), (0.21851460202031345,0.4615276820695843,0.8597927467636494), 0.5828191357596332, 0.6789132655854576, 0.7785035049413103,  (gDx*3, gDy*0, gDz*1)), rgb(0.2660811517569619,0.22989377258036298,0.5151311662347288)  );
  draw(  ellipsoid( (0.9459563802550541,0.1958268622733031,-0.2584924886083695), (-0.2608082402514035,0.9331184480943262,-0.24752580399425525), (0.19273190828641049,0.30156558463828315,0.9337626088519764), 0.6662991693193532, 0.7721046539568903, 0.8951656188953011,  (gDx*3, gDy*0, gDz*2)), rgb(0.26507859403937367,0.23316547899185436,0.5166959204405093)  );
  draw(  ellipsoid( (0.9420029390260659,0.2676696626676172,-0.20244360832996952), (-0.2937141798186659,0.9493748473491147,-0.11144227112856987), (-0.16236515463517165,-0.16443950531501986,-0.9729322718730469), 0.7391598630807297, 0.8449343311796323, 0.988647466529579,  (gDx*3, gDy*0, gDz*3)), rgb(0.26581465382532143,0.23076550867173595,0.5155490201345134)  );
  draw(  ellipsoid( (0.9724062181179277,-0.1929739634682515,-0.1310999480890176), (0.2022598765832687,0.41729583750912724,0.8859769332900699), (0.11626301772322166,0.8880457383608739,-0.4448119572234767), 0.4139754433377088, 0.4633962981709353, 0.5464573000772668,  (gDx*3, gDy*1, gDz*0)), rgb(0.2678955510561645,0.22379500278983888,0.5121301923469156)  );
  draw(  ellipsoid( (0.977609553610661,-0.09049407226809669,-0.18997469192895278), (0.2069698427038819,0.25053389000408943,0.9457252530045653), (0.03798743081987021,0.9638690745343763,-0.263653868274202), 0.4976606187031043, 0.6031249334682953, 0.6147128648814596,  (gDx*3, gDy*1, gDz*1)), rgb(0.27501716123132347,0.19569258937906506,0.4964802356178495)  );
  draw(  ellipsoid( (-0.9772762877197315,-0.0643672087151711,0.20196019385749786), (-0.09850386772119012,0.9815598941929162,-0.16382051811649645), (-0.18769135703251472,-0.17999176801789712,-0.9655956285842873), 0.5901056816978141, 0.6978052539089882, 0.7548964079840821,  (gDx*3, gDy*1, gDz*2)), rgb(0.2724026580332909,0.2069697322807893,0.5031105303528709)  );
  draw(  ellipsoid( (0.9546731231483627,0.23929189513044424,-0.1770271641953062), (-0.25297544724298276,0.965676978096708,-0.058918563002020405), (0.15685232236203717,0.10103149458901689,0.9824408308243022), 0.6783745625499105, 0.7847325128711464, 0.8968727005804309,  (gDx*3, gDy*1, gDz*3)), rgb(0.2678907830197439,0.22381119281552364,0.5121382341143944)  );
  draw(  ellipsoid( (0.4271073676205462,0.48317965205721447,0.7642752909535391), (-0.8745481580734205,0.43543842114676373,0.21344531055893762), (-0.22966239512518113,-0.7595596126882226,0.6085428325420345), 0.2821667523711556, 0.3233144661178671, 0.5168051423363188,  (gDx*3, gDy*2, gDz*0)), rgb(0.20121274707250655,0.38372713530986274,0.55429905031441)  );
  draw(  ellipsoid( (-0.8174457603951706,-0.14157647823444752,-0.5583354991603994), (-0.5240018994464073,0.5852718684200897,0.6187720496375173), (-0.23917449318128098,-0.7983814507156254,0.5526143510303462), 0.37014076395711576, 0.384016899573114, 0.5335612889655907,  (gDx*3, gDy*2, gDz*1)), rgb(0.24400884736733958,0.2899634589448419,0.5379107367975371)  );
  draw(  ellipsoid( (0.9722942648322982,-0.23129301222395476,-0.033873368161289795), (0.16086262177461466,0.5568851368856549,0.8148632776310654), (0.16960860676869452,0.7977358502738956,-0.5786626251087653), 0.4589872668082795, 0.5033745224944141, 0.5770513693914892,  (gDx*3, gDy*2, gDz*2)), rgb(0.2754769898257091,0.19352751560518366,0.4951484561949848)  );
  draw(  ellipsoid( (-0.9968435202063328,0.021563483120775086,0.07640688724419278), (-0.04184420091934353,-0.9605829938865363,-0.27482608083186), (0.0674689489474112,-0.27715578300240834,0.95845323979642), 0.558015354682847, 0.6247099930378331, 0.6721136509640796,  (gDx*3, gDy*2, gDz*3)), rgb(0.2803505756894167,0.16492558802070345,0.4759368631220056)  );
  draw(  ellipsoid( (0.315797728117498,0.5387301863507169,0.7810515867920305), (-0.9105149822194633,0.4036185536398525,0.08974703511294553), (-0.2668974748282898,-0.7395010814543788,0.6179837283117467), 0.16965870461020569, 0.2723235693558628, 0.5271244534071614,  (gDx*3, gDy*3, gDz*0)), rgb(0.14654379747221094,0.5144748257198766,0.5568806161378554)  );
  draw(  ellipsoid( (0.3531842077954369,0.46217407086774787,0.813422426283812), (-0.8709875529040385,0.479834459649076,0.10554418041518925), (-0.34152832686344475,-0.7457573462935606,0.5720178165049079), 0.20419503506164205, 0.2913940444757761, 0.5006717566879066,  (gDx*3, gDy*3, gDz*1)), rgb(0.16753916954015363,0.4614346595829215,0.5580963888184323)  );
  draw(  ellipsoid( (0.395274741540078,0.3848205504959874,0.834068955552471), (-0.8301706783265436,0.538305378991706,0.14506537765932384), (-0.39315966673511926,-0.7497602702628098,0.5322452569898285), 0.27145801297791705, 0.3292874374274058, 0.4920938027710571,  (gDx*3, gDy*3, gDz*2)), rgb(0.20569183480289832,0.37404828106100463,0.5533559059481891)  );
  draw(  ellipsoid( (0.501991381329083,0.31562060063632064,0.8052256140519173), (-0.764764252371535,0.5967931289267077,0.24284480550441517), (-0.4039062903272591,-0.7377137640845786,0.5409603598363143), 0.37506084869254214, 0.3981141105950489, 0.512995890300521,  (gDx*3, gDy*3, gDz*3)), rgb(0.2580598632222997,0.25406730628117924,0.5257425032823241)  );