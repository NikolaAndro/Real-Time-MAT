import numpy as np
import cvxpy as cp
import MAT_utils.utils_aux_MAT_attack as MAT_AUX
import matplotlib.pyplot as plt

loss_results = {
    'test_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.057482483337835, 1.6719873012955655, 1.7096302828837915, 1.7168439901981156, 1.7222167037196996, 1.6989352582656232, 1.7018874809914029, 1.7085919884062304, 1.7106656413717367, 1.7028018211581044],
                                    'normal': [0.3422231049267287, 0.35082457114740745, 0.3642564108076784, 0.36655474040926117, 0.36805894442440307, 0.3633676059467276, 0.3640569494557135, 0.3647810722749258, 0.36547634816661323, 0.3721145925448113]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0814998105629203, 1.4200549584074118, 1.5281260141392343, 1.6173551264497423, 1.6738585408200923, 1.6953587608239085, 1.702852010235344, 1.7062628645257851, 1.7062465896311494, 1.7115838281887095],
                                    'normal': [0.2922673722517859, 0.32525888030062017, 0.34725985871147863, 0.3688274564202299, 0.3623367377777689, 0.35893063062859565, 0.3663615887312545, 0.3745617415487152, 0.3768184171202256, 0.3781628898431345]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.1041493820160935, 1.6339177139026602, 1.65312754100131, 1.658142386515116, 1.6595340659937907, 1.6676589051472772, 1.670119223643824, 1.6716432060163047, 1.6745102690667222, 1.6767529200032814],
                                    'normal': [0.3300246169886638, 0.347245206599383, 0.3518711268287344, 0.352439781439673, 0.3526754198000603, 0.352919594651645, 0.35282778961142314, 0.35298536979046063, 0.3534351841690614, 0.3536302811706189]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.028387081991766, 1.547090113531683, 1.5949863578855377, 1.6158179941865587, 1.6619589891630349, 1.6655736714294276, 1.7004213153701468, 1.6975053145713412, 1.7024149761986487, 1.7125757532021435],
                                    'normal': [0.29395247510413536, 0.34281117913034775, 0.34176437375471763, 0.34511655390877083, 0.353580729617286, 0.3511239437098356, 0.3576238403308023, 0.3577928905941776, 0.35983746011847073, 0.3630908224693279]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.1218738142977056, 1.6677080380547906, 1.6997784592441676, 1.7105849054670825, 1.7138805996511401, 1.7069904000488754, 1.706759956694141, 1.703716875843166, 1.701049740044112, 1.7242404974612993],
                                    'normal': [0.3316511423931908, 0.37186298573140014, 0.3727930638593497, 0.37303979021986733, 0.37979782942644097, 0.3801705729101122, 0.3820594065582629, 0.38087092195589517, 0.38117558916204985, 0.37860409995329747]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.1337973383284106, 1.6092150909384502, 1.676007026003808, 1.6769315031385914, 1.676234197616577, 1.6797406727505713, 1.6836601579312196, 1.6928527974590812, 1.6994488755452264, 1.697665480240104],
                                    'normal': [0.30904505182172837, 0.3427108805511416, 0.3538515290034186, 0.37354462773529523, 0.37189136098340614, 0.3715857030804624, 0.37310867260411845, 0.37077624158760936, 0.373062776781849, 0.3704366102661054]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.0501654301722025, 1.7107831490408514, 1.710195165319541, 1.7412462158301443, 1.7510156255407432, 1.7246559612529795, 1.7581977165851397, 1.759352051105696, 1.7067319533259597, 1.776118222954347],
                                    'normal': [0.34013236158901883, 0.38763180706918854, 0.3890030801910715, 0.38685898750098713, 0.3881285353419707, 0.38424576090783186, 0.3918586657219326, 0.3905601308517849, 0.3849210178114704, 0.396578737263827]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.0565109294714388, 1.6303409817292518, 1.6649748524439703, 1.6941117586548795, 1.6914137761617445, 1.7045222545407481, 1.6970853286920136, 1.7028575614555594, 1.6676198357159329, 1.6900620249128833],
                                    'normal': [0.32295455938761997, 0.3665395888471112, 0.38640309062200723, 0.38812177531497993, 0.390170568965145, 0.38927188626269704, 0.3884876578739009, 0.39177207111083356, 0.3966857711678928, 0.39190258383750914]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0681745144509778, 1.6200553132086686, 1.582614915887105, 1.58298257724526, 1.6324817524742834, 1.563879829092124, 1.642745912935316, 1.613683144825021, 1.5944064096077202, 1.5875890768680376],
                                    'normal': [0.3082017257656019, 0.3512688435537299, 0.3648729525276066, 0.3623403733231358, 0.36340341457386605, 0.3614476731143047, 0.35875659606505916, 0.3686393425022204, 0.36666723704829657, 0.36154256739567237]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.0530458214356728, 1.4962609104274474, 1.5170351488074076, 1.5241774588516077, 1.5275923913287133, 1.5368895252955328, 1.5704784823447158, 1.5836862999139374, 1.5790066802624574, 1.5946923167435165],
                                    'normal': [0.31931462472247096, 0.3341914008265918, 0.33340500838977777, 0.33491001242829355, 0.33354593361775897, 0.3338856005791536, 0.3416630691474246, 0.34458429487710146, 0.34741318619128353, 0.34602430404461537]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [1.0575275857423998, 1.6722344781934602, 1.6782611298806889, 1.6836647807937308, 1.684944219441758, 1.6526634339204769, 1.6701115787643748, 1.682510008517, 1.6803147923086108, 1.7036585788136904],
                                    'normal': [0.28634685987664255, 0.347442101725598, 0.34790225932278585, 0.34748688819482154, 0.35177282144113914, 0.3510158755115627, 0.35173952797639, 0.3485190726432604, 0.3502403815382535, 0.3528551603408204]
                                },
                                'POF-based-greedy':        {
                                    'zdepth': [1.1049068434951232, 1.563132699986094, 1.600042385661725, 1.6126589664478892, 1.6505921602249145, 1.6753942610062276, 1.6951732610918813, 1.6992259789987938, 1.7373498361135267, 1.727946020893215],
                                    'normal': [0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055]
                                },
           }, 
           '3_points_3_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [1.0197926274279958, 1.584934410606463, 1.6194321379219134, 1.5813500286377582, 1.600242920757569, 1.6217247053519968, 1.5946104885376606, 1.5979767413483452, 1.5971017984999825, 1.5926305060534134],
                                    'normal': [0.3124103307109518, 0.35754515774471246, 0.3708703393174201, 0.3810851670417589, 0.368305677974347, 0.3718360502080819, 0.37477046171414485, 0.3740251626550537, 0.3749897020993773, 0.3758367668107613]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [1.1509047252615703, 1.5574920654296875, 1.6228267753247134, 1.627018292171439, 1.6256316720824882, 1.6197604584939702, 1.627216966864989, 1.6308016796701963, 1.6459725819912154, 1.6540683006502919],
                                    'normal': [0.31330102153660094, 0.3208603039844749, 0.347156644819938, 0.3482491232685207, 0.34852160633224805, 0.35474043927856325, 0.3536826352175978, 0.3579348233557239, 0.3711348551143076, 0.3722422357072535]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.9932249545436544, 1.4832410564127656, 1.489613993880675, 1.4889670389214742, 1.4953558169689376, 1.4957027415639346, 1.4996933678990787, 1.5114310173644232, 1.5361213551354163, 1.5571300538544803],
                                    'normal': [0.2923094980495492, 0.30608028867810044, 0.3061699933305229, 0.30667862609489677, 0.30865949917699875, 0.30906977960743853, 0.3094678116213415, 0.3103248262221051, 0.31615788180803517, 0.32675274002183347]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [1.0906920554097166, 1.5517956967206346, 1.5837280379128211, 1.6299365930950518, 1.6280626432182863, 1.6428285289056523, 1.6413225739272599, 1.6502504078383298, 1.6550571257306128, 1.6616724928629767],
                                    'normal': [0.3128962102624559, 0.3412199582328501, 0.35081258830950435, 0.37090757691983095, 0.37304771014095583, 0.3742851836779683, 0.37022044228524276, 0.3741223335880594, 0.3746446664185868, 0.3750189649075577]
                                },
           }, 
        },
    'clean_results': {
           '2_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817, 0.7930933188531817],
                                    'normal': [0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886, 0.34809870873529886]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946, 0.5815829763092946],
                                    'normal': [0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827, 0.33575111833429827]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873, 0.7819748673856873],
                                    'normal': [0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431, 0.3571154062895431]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069, 0.7088344784127069],
                                    'normal': [0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574, 0.31159852300722574]
                                },
           }, 
           '2_points_5_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985, 0.9282419395200985],
                                    'normal': [0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255, 0.36543824728002255]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066, 0.6887274433657066],
                                    'normal': [0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357, 0.32900340815180357]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347, 0.7011490742570347],
                                    'normal': [0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736, 0.3622551855352736]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184, 0.5759699804881184],
                                    'normal': [0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578, 0.3567218330717578]
                                },
           }, 
           '5_points_2_predators' : {
                                'antipodes-non-greedy' :  {
                                    'zdepth': [0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212, 0.6391964393178212],
                                    'normal': [0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033, 0.3516883749630033]
                                },
                                'antipodes-greedy':           {
                                    'zdepth': [0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754, 0.6004219682560754],
                                    'normal': [0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873, 0.3356046881257873]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396, 0.4817260866005396],
                                    'normal': [0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428, 0.3211764942739428]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905, 0.7868108901780905],
                                    'normal': [0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055, 0.27768460936153055]
                                },
           }, 
           '3_points_3_predators' : {
                            'antipodes-non-greedy' :       {
                                    'zdepth': [0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761, 0.4879108712230761],
                                    'normal': [0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607, 0.33998285903758607]
                                },
                                'antipodes-greedy':       {
                                    'zdepth': [0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143, 0.6285741151608143],
                                    'normal': [0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046, 0.3406812053673046]
                                },
                                'POF-based-non-greedy':   {
                                    'zdepth': [0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678, 0.5297742498289678],
                                    'normal': [0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686, 0.30875081085052686]
                                },
                                'POF-based-greedy':       {
                                    'zdepth': [0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143, 0.6852263158744143],
                                    'normal': [0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157, 0.2975890907430157]
                                },
           }, 
        },
    
}

def fitness_evaluation_validation(loss_x,loss_y, A):
    '''
    Description: 
        - fitness evaluation in the validation phase
    Arguments:
        - losses: dictionary of losses
        - A: matrix representing factors of variables of POF equation
        - b: RHS of the POF equation
    Returns:
        - fitness based on the losses provided. 
    '''           
    point_in_loss_space = [loss_x,loss_y]

    # Problem dimensions (m inequalities in n-dimensional space).
    n = len(point_in_loss_space)


    x_l2 = cp.Variable(shape=n, pos=True)
    constraints = [x_l2[0] >= 0.1, cp.inv_prod(x_l2) <= np.prod(A)]

    # Form objective.
    obj = cp.Minimize(cp.norm(x_l2 - point_in_loss_space, 2))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return obj.value





A = np.array([[1.8,1.0]])

# fitness = []
# for zdepth, normal in zip(loss_results['test_results']['2_points_5_predators']['antipodes-non-greedy']['zdepth'],loss_results['test_results']['2_points_5_predators']['antipodes-non-greedy']['normal']):
#     fitness.append(fitness_evaluation_validation(zdepth, normal,A))

fitness=[]
for zdepth, normal in zip(loss_results['test_results']['2_points_2_predators']['POF-based-greedy']['zdepth'],loss_results['test_results']['2_points_2_predators']['POF-based-greedy']['normal']):
    fitness.append(fitness_evaluation_validation(zdepth, normal,A))
minimum = min(fitness)
index = fitness.index(minimum)

print("The index of the maximum fitness of 2x2 23G scenario is ", index, " and its fitness is ", minimum)
print(fitness)
# plt.plot(loss_results['test_results']['2_points_2_predators']['POF-based-greedy']['zdepth'][index],loss_results['test_results']['2_points_2_predators']['POF-based-greedy']['normal'][index],'*', color='cyan' ,label='POF-based-greedy')


# epochs = range(1,21)
# plt.figure(1)
# plt.plot(epochs,fitness,'-', color='red' ,label='antipodes-non-greedy')


# plt.xticks(epochs[::2])
# plt.xlabel("Epoch")
# plt.ylabel("Fitness")
# # plt.title("Depth zbuffer" + " VS. " + "Surface normal" + " --Testing Fitness ")
# plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# plt.show()
# plt.close(1)