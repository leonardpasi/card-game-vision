йш
 Ѓ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12unknown8ѕн
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	Р@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@$*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:$*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:$*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:$*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:$*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:$*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:$*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
З
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*&
shared_nameAdam/dense_3/kernel/m
А
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	Р@*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@$*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:$*
dtype0
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:$*
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:$*
dtype0
Ж
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:$*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
З
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*&
shared_nameAdam/dense_3/kernel/v
А
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	Р@*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@$*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@$*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:$*
dtype0
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:$*
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:$*
dtype0
Ж
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:$*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ьA
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЈA
value≠AB™A B£A
ќ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Ч
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
Ч
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
ъ
:iter

;beta_1

<beta_2
	=decay
>learning_ratemrmsmtmu!mv"mw(mx)my4mz5m{v|v}v~v!vА"vБ(vВ)vГ4vД5vЕ
f
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
412
513
F
0
1
2
3
!4
"5
(6
)7
48
59
 
≠
?metrics
		variables

trainable_variables
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses

Clayers
 
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
Dmetrics
	variables
trainable_variables
regularization_losses
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables

Hlayers
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
≠
Imetrics
	variables
trainable_variables
regularization_losses
Jlayer_metrics
Klayer_regularization_losses
Lnon_trainable_variables

Mlayers
 
 
 
≠
Nmetrics
	variables
trainable_variables
regularization_losses
Olayer_metrics
Player_regularization_losses
Qnon_trainable_variables

Rlayers
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
≠
Smetrics
#	variables
$trainable_variables
%regularization_losses
Tlayer_metrics
Ulayer_regularization_losses
Vnon_trainable_variables

Wlayers
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
*2
+3

(0
)1
 
≠
Xmetrics
,	variables
-trainable_variables
.regularization_losses
Ylayer_metrics
Zlayer_regularization_losses
[non_trainable_variables

\layers
 
 
 
≠
]metrics
0	variables
1trainable_variables
2regularization_losses
^layer_metrics
_layer_regularization_losses
`non_trainable_variables

alayers
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
≠
bmetrics
6	variables
7trainable_variables
8regularization_losses
clayer_metrics
dlayer_regularization_losses
enon_trainable_variables

flayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
 
 

0
1
*2
+3
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 
 
 
4
	itotal
	jcount
k	variables
l	keras_api
D
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

p	variables
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_2Placeholder*(
_output_shapes
:€€€€€€€€€Р*
dtype0*
shape:€€€€€€€€€Р
Ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_3/kerneldense_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_4/kerneldense_4/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_201771
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Л
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_202431
™

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_4/kerneldense_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_3/kernel/mAdam/dense_3/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_4/kernel/mAdam/dense_4/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_3/kernel/vAdam/dense_3/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_4/kernel/vAdam/dense_4/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_202570у±

µ
©
6__inference_batch_normalization_2_layer_call_fn_202090

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011442
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
‘%
ґ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201583
input_2
dense_3_201547
dense_3_201549 
batch_normalization_2_201552 
batch_normalization_2_201554 
batch_normalization_2_201556 
batch_normalization_2_201558
dense_4_201562
dense_4_201564 
batch_normalization_3_201567 
batch_normalization_3_201569 
batch_normalization_3_201571 
batch_normalization_3_201573
dense_5_201577
dense_5_201579
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallР
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_201547dense_3_201549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2013432!
dense_3/StatefulPartitionedCallЈ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_2_201552batch_normalization_2_201554batch_normalization_2_201556batch_normalization_2_201558*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011772/
-batch_normalization_2/StatefulPartitionedCallЗ
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014112
dropout_2/PartitionedCallЂ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_201562dense_4_201564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2014352!
dense_4/StatefulPartitionedCallЈ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_201567batch_normalization_3_201569batch_normalization_3_201571batch_normalization_3_201573*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2013172/
-batch_normalization_3/StatefulPartitionedCallЗ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2015032
dropout_3/PartitionedCallЂ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_201577dense_5_201579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2015272!
dense_5/StatefulPartitionedCall¬
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
†
c
*__inference_dropout_3_layer_call_fn_202254

inputs
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2014982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€$22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
‘
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_201317

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
м	
Ї
-__inference_sequential_1_layer_call_fn_201656
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2016252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
Б
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_202244

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€$*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€$2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€$:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
х	
№
C__inference_dense_5_layer_call_and_return_conditional_losses_202270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
‘
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_201177

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Џ
}
(__inference_dense_5_layer_call_fn_202279

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2015272
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
х	
№
C__inference_dense_5_layer_call_and_return_conditional_losses_201527

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
ј	
±
$__inference_signature_wrapper_201771
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_2010482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
»
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_201411

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
н	
є
-__inference_sequential_1_layer_call_fn_202001

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2016972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ЬЇ
Э
H__inference_sequential_1_layer_call_and_return_conditional_losses_201876

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource0
,batch_normalization_2_assignmovingavg_2017892
.batch_normalization_2_assignmovingavg_1_201795?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource0
,batch_normalization_3_assignmovingavg_2018362
.batch_normalization_3_assignmovingavg_1_201842?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИҐ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOp¶
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/Reluґ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesе
"batch_normalization_2/moments/meanMeandense_3/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_2/moments/meanЊ
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_2/moments/StopGradientъ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_3/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/batch_normalization_2/moments/SquaredDifferenceЊ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesК
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_2/moments/variance¬
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze 
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1О
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/201789*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_2/AssignMovingAvg/decay’
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_201789*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpя
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/201789*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/sub÷
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/201789*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul≥
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_201789-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/201789*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/201795*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_2/AssignMovingAvg_1/decayџ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_201795*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/201795*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/subа
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/201795*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulњ
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_201795/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/201795*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yЏ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mulћ
%batch_normalization_2/batchnorm/mul_1Muldense_3/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/mul_1”
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpў
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subЁ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1w
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_2/dropout/Constі
dropout_2/dropout/MulMul)batch_normalization_2/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_2/dropout/MulЛ
dropout_2/dropout/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape“
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_2/dropout/GreaterEqual/yж
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_2/dropout/CastҐ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_2/dropout/Mul_1•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/Reluґ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesе
"batch_normalization_3/moments/meanMeandense_4/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2$
"batch_normalization_3/moments/meanЊ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:$2,
*batch_normalization_3/moments/StopGradientъ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_4/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€$21
/batch_normalization_3/moments/SquaredDifferenceЊ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesК
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2(
&batch_normalization_3/moments/variance¬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze 
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1О
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/201836*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_3/AssignMovingAvg/decay’
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_201836*
_output_shapes
:$*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpя
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/201836*
_output_shapes
:$2+
)batch_normalization_3/AssignMovingAvg/sub÷
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/201836*
_output_shapes
:$2+
)batch_normalization_3/AssignMovingAvg/mul≥
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_201836-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/201836*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/201842*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_3/AssignMovingAvg_1/decayџ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_201842*
_output_shapes
:$*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/201842*
_output_shapes
:$2-
+batch_normalization_3/AssignMovingAvg_1/subа
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/201842*
_output_shapes
:$2-
+batch_normalization_3/AssignMovingAvg_1/mulњ
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_201842/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/201842*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yЏ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:$2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/mulћ
%batch_normalization_3/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:$2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2'
%batch_normalization_3/batchnorm/add_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout_3/dropout/Constі
dropout_3/dropout/MulMul)batch_normalization_3/batchnorm/add_1:z:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout_3/dropout/MulЛ
dropout_3/dropout/ShapeShape)batch_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape“
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€$*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 dropout_3/dropout/GreaterEqual/yж
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2 
dropout_3/dropout/GreaterEqualЭ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€$2
dropout_3/dropout/CastҐ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout_3/dropout/Mul_1•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
dense_5/MatMul/ReadVariableOp†
dense_5/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/Softmax–
IdentityIdentitydense_5/Softmax:softmax:0:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
р	
№
C__inference_dense_3_layer_call_and_return_conditional_losses_202012

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Ј
©
6__inference_batch_normalization_3_layer_call_fn_202232

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2013172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
»
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_202120

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
‘
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202077

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
р	
№
C__inference_dense_3_layer_call_and_return_conditional_losses_201343

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Ф
F
*__inference_dropout_2_layer_call_fn_202130

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014112
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ф
F
*__inference_dropout_3_layer_call_fn_202259

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2015032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€$:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
†
c
*__inference_dropout_2_layer_call_fn_202125

inputs
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
З0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_201284

inputs
assignmovingavg_201259
assignmovingavg_1_201265)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:$2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/201259*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201259*
_output_shapes
:$*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/201259*
_output_shapes
:$2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/201259*
_output_shapes
:$2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201259AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/201259*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/201265*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201265*
_output_shapes
:$*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201265*
_output_shapes
:$2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201265*
_output_shapes
:$2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201265AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/201265*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
вЈ
‘
"__inference__traced_restore_202570
file_prefix#
assignvariableop_dense_3_kernel#
assignvariableop_1_dense_3_bias2
.assignvariableop_2_batch_normalization_2_gamma1
-assignvariableop_3_batch_normalization_2_beta8
4assignvariableop_4_batch_normalization_2_moving_mean<
8assignvariableop_5_batch_normalization_2_moving_variance%
!assignvariableop_6_dense_4_kernel#
assignvariableop_7_dense_4_bias2
.assignvariableop_8_batch_normalization_3_gamma1
-assignvariableop_9_batch_normalization_3_beta9
5assignvariableop_10_batch_normalization_3_moving_mean=
9assignvariableop_11_batch_normalization_3_moving_variance&
"assignvariableop_12_dense_5_kernel$
 assignvariableop_13_dense_5_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1-
)assignvariableop_23_adam_dense_3_kernel_m+
'assignvariableop_24_adam_dense_3_bias_m:
6assignvariableop_25_adam_batch_normalization_2_gamma_m9
5assignvariableop_26_adam_batch_normalization_2_beta_m-
)assignvariableop_27_adam_dense_4_kernel_m+
'assignvariableop_28_adam_dense_4_bias_m:
6assignvariableop_29_adam_batch_normalization_3_gamma_m9
5assignvariableop_30_adam_batch_normalization_3_beta_m-
)assignvariableop_31_adam_dense_5_kernel_m+
'assignvariableop_32_adam_dense_5_bias_m-
)assignvariableop_33_adam_dense_3_kernel_v+
'assignvariableop_34_adam_dense_3_bias_v:
6assignvariableop_35_adam_batch_normalization_2_gamma_v9
5assignvariableop_36_adam_batch_normalization_2_beta_v-
)assignvariableop_37_adam_dense_4_kernel_v+
'assignvariableop_38_adam_dense_4_bias_v:
6assignvariableop_39_adam_batch_normalization_3_gamma_v9
5assignvariableop_40_adam_batch_normalization_3_beta_v-
)assignvariableop_41_adam_dense_5_kernel_v+
'assignvariableop_42_adam_dense_5_bias_v
identity_44ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ь
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*И
valueюBы,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesж
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesК
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*∆
_output_shapes≥
∞::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≥
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_2_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3≤
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_2_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4є
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_2_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5љ
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_2_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≥
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9≤
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10љ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ѕ
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12™
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14•
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15І
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16І
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19°
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ѓ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Њ
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_batch_normalization_2_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26љ
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_batch_normalization_2_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27±
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ѓ
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Њ
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_batch_normalization_3_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30љ
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_batch_normalization_3_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ѓ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ѓ
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Њ
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_2_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36љ
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_2_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ѓ
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Њ
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_3_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40љ
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_3_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpР
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43Г
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*√
_input_shapes±
Ѓ: :::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
‘(
ю
H__inference_sequential_1_layer_call_and_return_conditional_losses_201544
input_2
dense_3_201354
dense_3_201356 
batch_normalization_2_201385 
batch_normalization_2_201387 
batch_normalization_2_201389 
batch_normalization_2_201391
dense_4_201446
dense_4_201448 
batch_normalization_3_201477 
batch_normalization_3_201479 
batch_normalization_3_201481 
batch_normalization_3_201483
dense_5_201538
dense_5_201540
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallР
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_201354dense_3_201356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2013432!
dense_3/StatefulPartitionedCallµ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_2_201385batch_normalization_2_201387batch_normalization_2_201389batch_normalization_2_201391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011442/
-batch_normalization_2/StatefulPartitionedCallЯ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014062#
!dropout_2/StatefulPartitionedCall≥
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_201446dense_4_201448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2014352!
dense_4/StatefulPartitionedCallµ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_201477batch_normalization_3_201479batch_normalization_3_201481batch_normalization_3_201483*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2012842/
-batch_normalization_3/StatefulPartitionedCall√
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2014982#
!dropout_3/StatefulPartitionedCall≥
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_201538dense_5_201540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2015272!
dense_5/StatefulPartitionedCallК
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
р	
Ї
-__inference_sequential_1_layer_call_fn_201728
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2016972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
А[
К
__inference__traced_save_202431
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*И
valueюBы,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesа
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices„
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ю
_input_shapesМ
Й: :	Р@:@:@:@:@:@:@$:$:$:$:$:$:$:: : : : : : : : : :	Р@:@:@:@:@$:$:$:$:$::	Р@:@:@:@:@$:$:$:$:$:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Р@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@$: 

_output_shapes
:$: 	

_output_shapes
:$: 


_output_shapes
:$: 

_output_shapes
:$: 

_output_shapes
:$:$ 

_output_shapes

:$: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Р@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@$: 

_output_shapes
:$: 

_output_shapes
:$: 

_output_shapes
:$:$  

_output_shapes

:$: !

_output_shapes
::%"!

_output_shapes
:	Р@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:$& 

_output_shapes

:@$: '

_output_shapes
:$: (

_output_shapes
:$: )

_output_shapes
:$:$* 

_output_shapes

:$: +

_output_shapes
::,

_output_shapes
: 
—%
µ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201697

inputs
dense_3_201661
dense_3_201663 
batch_normalization_2_201666 
batch_normalization_2_201668 
batch_normalization_2_201670 
batch_normalization_2_201672
dense_4_201676
dense_4_201678 
batch_normalization_3_201681 
batch_normalization_3_201683 
batch_normalization_3_201685 
batch_normalization_3_201687
dense_5_201691
dense_5_201693
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallП
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_201661dense_3_201663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2013432!
dense_3/StatefulPartitionedCallЈ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_2_201666batch_normalization_2_201668batch_normalization_2_201670batch_normalization_2_201672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011772/
-batch_normalization_2/StatefulPartitionedCallЗ
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014112
dropout_2/PartitionedCallЂ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_201676dense_4_201678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2014352!
dense_4/StatefulPartitionedCallЈ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_201681batch_normalization_3_201683batch_normalization_3_201685batch_normalization_3_201687*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2013172/
-batch_normalization_3/StatefulPartitionedCallЗ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2015032
dropout_3/PartitionedCallЂ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_201691dense_5_201693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2015272!
dense_5/StatefulPartitionedCall¬
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
н	
№
C__inference_dense_4_layer_call_and_return_conditional_losses_202141

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ФX
≈
H__inference_sequential_1_layer_call_and_return_conditional_losses_201935

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOp¶
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_3/Relu‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yа
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mulћ
%batch_normalization_2/batchnorm/mul_1Muldense_3/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/mul_1Џ
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Ё
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2Џ
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2џ
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subЁ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%batch_normalization_2/batchnorm/add_1С
dropout_2/IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_2/Identity•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dense_4/Relu‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yа
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:$2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/mulћ
%batch_normalization_3/batchnorm/mul_1Muldense_4/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:$2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2%
#batch_normalization_3/batchnorm/subЁ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2'
%batch_normalization_3/batchnorm/add_1С
dropout_3/IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout_3/Identity•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02
dense_5/MatMul/ReadVariableOp†
dense_5/MatMulMatMuldropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/Softmax»
IdentityIdentitydense_5/Softmax:softmax:0/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
З0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202057

inputs
assignmovingavg_202032
assignmovingavg_1_202038)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/202032*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_202032*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/202032*
_output_shapes
:@2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/202032*
_output_shapes
:@2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_202032AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/202032*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/202038*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_202038*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202038*
_output_shapes
:@2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202038*
_output_shapes
:@2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_202038AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/202038*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
н	
№
C__inference_dense_4_layer_call_and_return_conditional_losses_201435

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Тm
Л
!__inference__wrapped_model_201048
input_27
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resourceH
Dsequential_1_batch_normalization_2_batchnorm_readvariableop_resourceL
Hsequential_1_batch_normalization_2_batchnorm_mul_readvariableop_resourceJ
Fsequential_1_batch_normalization_2_batchnorm_readvariableop_1_resourceJ
Fsequential_1_batch_normalization_2_batchnorm_readvariableop_2_resource7
3sequential_1_dense_4_matmul_readvariableop_resource8
4sequential_1_dense_4_biasadd_readvariableop_resourceH
Dsequential_1_batch_normalization_3_batchnorm_readvariableop_resourceL
Hsequential_1_batch_normalization_3_batchnorm_mul_readvariableop_resourceJ
Fsequential_1_batch_normalization_3_batchnorm_readvariableop_1_resourceJ
Fsequential_1_batch_normalization_3_batchnorm_readvariableop_2_resource7
3sequential_1_dense_5_matmul_readvariableop_resource8
4sequential_1_dense_5_biasadd_readvariableop_resource
identityИҐ;sequential_1/batch_normalization_2/batchnorm/ReadVariableOpҐ=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2Ґ?sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ;sequential_1/batch_normalization_3/batchnorm/ReadVariableOpҐ=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ?sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ+sequential_1/dense_3/BiasAdd/ReadVariableOpҐ*sequential_1/dense_3/MatMul/ReadVariableOpҐ+sequential_1/dense_4/BiasAdd/ReadVariableOpҐ*sequential_1/dense_4/MatMul/ReadVariableOpҐ+sequential_1/dense_5/BiasAdd/ReadVariableOpҐ*sequential_1/dense_5/MatMul/ReadVariableOpЌ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp≥
sequential_1/dense_3/MatMulMatMulinput_22sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_3/MatMulЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp’
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_3/BiasAddЧ
sequential_1/dense_3/ReluRelu%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_3/Reluы
;sequential_1/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpDsequential_1_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential_1/batch_normalization_2/batchnorm/ReadVariableOp≠
2sequential_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2sequential_1/batch_normalization_2/batchnorm/add/yФ
0sequential_1/batch_normalization_2/batchnorm/addAddV2Csequential_1/batch_normalization_2/batchnorm/ReadVariableOp:value:0;sequential_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_2/batchnorm/addћ
2sequential_1/batch_normalization_2/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@24
2sequential_1/batch_normalization_2/batchnorm/RsqrtЗ
?sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02A
?sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOpС
0sequential_1/batch_normalization_2/batchnorm/mulMul6sequential_1/batch_normalization_2/batchnorm/Rsqrt:y:0Gsequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_2/batchnorm/mulА
2sequential_1/batch_normalization_2/batchnorm/mul_1Mul'sequential_1/dense_3/Relu:activations:04sequential_1/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@24
2sequential_1/batch_normalization_2/batchnorm/mul_1Б
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_1_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1С
2sequential_1/batch_normalization_2/batchnorm/mul_2MulEsequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:04sequential_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@24
2sequential_1/batch_normalization_2/batchnorm/mul_2Б
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_1_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02?
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2П
0sequential_1/batch_normalization_2/batchnorm/subSubEsequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:06sequential_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_2/batchnorm/subС
2sequential_1/batch_normalization_2/batchnorm/add_1AddV26sequential_1/batch_normalization_2/batchnorm/mul_1:z:04sequential_1/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@24
2sequential_1/batch_normalization_2/batchnorm/add_1Є
sequential_1/dropout_2/IdentityIdentity6sequential_1/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
sequential_1/dropout_2/Identityћ
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:@$*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOp‘
sequential_1/dense_4/MatMulMatMul(sequential_1/dropout_2/Identity:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
sequential_1/dense_4/MatMulЋ
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOp’
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
sequential_1/dense_4/BiasAddЧ
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
sequential_1/dense_4/Reluы
;sequential_1/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpDsequential_1_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype02=
;sequential_1/batch_normalization_3/batchnorm/ReadVariableOp≠
2sequential_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:24
2sequential_1/batch_normalization_3/batchnorm/add/yФ
0sequential_1/batch_normalization_3/batchnorm/addAddV2Csequential_1/batch_normalization_3/batchnorm/ReadVariableOp:value:0;sequential_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:$22
0sequential_1/batch_normalization_3/batchnorm/addћ
2sequential_1/batch_normalization_3/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:$24
2sequential_1/batch_normalization_3/batchnorm/RsqrtЗ
?sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype02A
?sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOpС
0sequential_1/batch_normalization_3/batchnorm/mulMul6sequential_1/batch_normalization_3/batchnorm/Rsqrt:y:0Gsequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$22
0sequential_1/batch_normalization_3/batchnorm/mulА
2sequential_1/batch_normalization_3/batchnorm/mul_1Mul'sequential_1/dense_4/Relu:activations:04sequential_1/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$24
2sequential_1/batch_normalization_3/batchnorm/mul_1Б
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype02?
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1С
2sequential_1/batch_normalization_3/batchnorm/mul_2MulEsequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:04sequential_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:$24
2sequential_1/batch_normalization_3/batchnorm/mul_2Б
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype02?
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2П
0sequential_1/batch_normalization_3/batchnorm/subSubEsequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:06sequential_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$22
0sequential_1/batch_normalization_3/batchnorm/subС
2sequential_1/batch_normalization_3/batchnorm/add_1AddV26sequential_1/batch_normalization_3/batchnorm/mul_1:z:04sequential_1/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$24
2sequential_1/batch_normalization_3/batchnorm/add_1Є
sequential_1/dropout_3/IdentityIdentity6sequential_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2!
sequential_1/dropout_3/Identityћ
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOp‘
sequential_1/dense_5/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_5/MatMulЋ
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOp’
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_5/BiasAdd†
sequential_1/dense_5/SoftmaxSoftmax%sequential_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_5/SoftmaxЛ
IdentityIdentity&sequential_1/dense_5/Softmax:softmax:0<^sequential_1/batch_normalization_2/batchnorm/ReadVariableOp>^sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1>^sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2@^sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOp<^sequential_1/batch_normalization_3/batchnorm/ReadVariableOp>^sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1>^sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2@^sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2z
;sequential_1/batch_normalization_2/batchnorm/ReadVariableOp;sequential_1/batch_normalization_2/batchnorm/ReadVariableOp2~
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_1=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_12~
=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_2=sequential_1/batch_normalization_2/batchnorm/ReadVariableOp_22В
?sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOp?sequential_1/batch_normalization_2/batchnorm/mul/ReadVariableOp2z
;sequential_1/batch_normalization_3/batchnorm/ReadVariableOp;sequential_1/batch_normalization_3/batchnorm/ReadVariableOp2~
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_1=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_12~
=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_2=sequential_1/batch_normalization_3/batchnorm/ReadVariableOp_22В
?sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOp?sequential_1/batch_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp:Q M
(
_output_shapes
:€€€€€€€€€Р
!
_user_specified_name	input_2
й	
є
-__inference_sequential_1_layer_call_fn_201968

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_2016252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
З0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202186

inputs
assignmovingavg_202161
assignmovingavg_1_202167)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:$2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/202161*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_202161*
_output_shapes
:$*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/202161*
_output_shapes
:$2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/202161*
_output_shapes
:$2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_202161AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/202161*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/202167*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_202167*
_output_shapes
:$*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202167*
_output_shapes
:$2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202167*
_output_shapes
:$2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_202167AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/202167*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
Џ
}
(__inference_dense_4_layer_call_fn_202150

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2014352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
µ
©
6__inference_batch_normalization_3_layer_call_fn_202219

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2012842
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
‘
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202206

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
»
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_201503

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€$2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€$:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
Ј
©
6__inference_batch_normalization_2_layer_call_fn_202103

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011772
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Б
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_201406

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
З0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_201144

inputs
assignmovingavg_201119
assignmovingavg_1_201125)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/201119*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201119*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/201119*
_output_shapes
:@2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/201119*
_output_shapes
:@2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201119AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/201119*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/201125*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201125*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201125*
_output_shapes
:@2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201125*
_output_shapes
:@2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201125AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/201125*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Б
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_201498

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€$*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€$2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€$2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€$:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs
—(
э
H__inference_sequential_1_layer_call_and_return_conditional_losses_201625

inputs
dense_3_201589
dense_3_201591 
batch_normalization_2_201594 
batch_normalization_2_201596 
batch_normalization_2_201598 
batch_normalization_2_201600
dense_4_201604
dense_4_201606 
batch_normalization_3_201609 
batch_normalization_3_201611 
batch_normalization_3_201613 
batch_normalization_3_201615
dense_5_201619
dense_5_201621
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallП
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_201589dense_3_201591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2013432!
dense_3/StatefulPartitionedCallµ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_2_201594batch_normalization_2_201596batch_normalization_2_201598batch_normalization_2_201600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2011442/
-batch_normalization_2/StatefulPartitionedCallЯ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2014062#
!dropout_2/StatefulPartitionedCall≥
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_201604dense_4_201606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2014352!
dense_4/StatefulPartitionedCallµ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_3_201609batch_normalization_3_201611batch_normalization_3_201613batch_normalization_3_201615*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2012842/
-batch_normalization_3/StatefulPartitionedCall√
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2014982#
!dropout_3/StatefulPartitionedCall≥
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_201619dense_5_201621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2015272!
dense_5/StatefulPartitionedCallК
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€Р::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
№
}
(__inference_dense_3_layer_call_fn_202021

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2013432
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Б
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_202115

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
»
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_202249

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€$2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€$:O K
'
_output_shapes
:€€€€€€€€€$
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
<
input_21
serving_default_input_2:0€€€€€€€€€Р;
dense_50
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:иш
Ь9
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
+Ж&call_and_return_all_conditional_losses
З__call__
И_default_save_signature"с5
_tf_keras_sequential“5{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 13, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 13, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.10000000149011612, "decay": 2.499999936844688e-05, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ф

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
і	
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
з
	variables
trainable_variables
regularization_losses
 	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
т

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 36, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
і	
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
з
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
х

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 13, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
Н
:iter

;beta_1

<beta_2
	=decay
>learning_ratemrmsmtmu!mv"mw(mx)my4mz5m{v|v}v~v!vА"vБ(vВ)vГ4vД5vЕ"
	optimizer
Ж
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
412
513"
trackable_list_wrapper
f
0
1
2
3
!4
"5
(6
)7
48
59"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
?metrics
		variables

trainable_variables
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses

Clayers
З__call__
И_default_save_signature
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map
!:	Р@2dense_3/kernel
:@2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Dmetrics
	variables
trainable_variables
regularization_losses
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables

Hlayers
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Imetrics
	variables
trainable_variables
regularization_losses
Jlayer_metrics
Klayer_regularization_losses
Lnon_trainable_variables

Mlayers
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Nmetrics
	variables
trainable_variables
regularization_losses
Olayer_metrics
Player_regularization_losses
Qnon_trainable_variables

Rlayers
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 :@$2dense_4/kernel
:$2dense_4/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Smetrics
#	variables
$trainable_variables
%regularization_losses
Tlayer_metrics
Ulayer_regularization_losses
Vnon_trainable_variables

Wlayers
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'$2batch_normalization_3/gamma
(:&$2batch_normalization_3/beta
1:/$ (2!batch_normalization_3/moving_mean
5:3$ (2%batch_normalization_3/moving_variance
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Xmetrics
,	variables
-trainable_variables
.regularization_losses
Ylayer_metrics
Zlayer_regularization_losses
[non_trainable_variables

\layers
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
]metrics
0	variables
1trainable_variables
2regularization_losses
^layer_metrics
_layer_regularization_losses
`non_trainable_variables

alayers
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 :$2dense_5/kernel
:2dense_5/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
bmetrics
6	variables
7trainable_variables
8regularization_losses
clayer_metrics
dlayer_regularization_losses
enon_trainable_variables

flayers
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
g0
h1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
*2
+3"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ї
	itotal
	jcount
k	variables
l	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ч
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api"–
_tf_keras_metricµ{"class_name": "MeanMetricWrapper", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
&:$	Р@2Adam/dense_3/kernel/m
:@2Adam/dense_3/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
%:#@$2Adam/dense_4/kernel/m
:$2Adam/dense_4/bias/m
.:,$2"Adam/batch_normalization_3/gamma/m
-:+$2!Adam/batch_normalization_3/beta/m
%:#$2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
&:$	Р@2Adam/dense_3/kernel/v
:@2Adam/dense_3/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
%:#@$2Adam/dense_4/kernel/v
:$2Adam/dense_4/bias/v
.:,$2"Adam/batch_normalization_3/gamma/v
-:+$2!Adam/batch_normalization_3/beta/v
%:#$2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
о2л
H__inference_sequential_1_layer_call_and_return_conditional_losses_201876
H__inference_sequential_1_layer_call_and_return_conditional_losses_201935
H__inference_sequential_1_layer_call_and_return_conditional_losses_201583
H__inference_sequential_1_layer_call_and_return_conditional_losses_201544ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
-__inference_sequential_1_layer_call_fn_202001
-__inference_sequential_1_layer_call_fn_201968
-__inference_sequential_1_layer_call_fn_201656
-__inference_sequential_1_layer_call_fn_201728ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
а2Ё
!__inference__wrapped_model_201048Ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *'Ґ$
"К
input_2€€€€€€€€€Р
н2к
C__inference_dense_3_layer_call_and_return_conditional_losses_202012Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_3_layer_call_fn_202021Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202057
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202077і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
6__inference_batch_normalization_2_layer_call_fn_202090
6__inference_batch_normalization_2_layer_call_fn_202103і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_2_layer_call_and_return_conditional_losses_202115
E__inference_dropout_2_layer_call_and_return_conditional_losses_202120і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_2_layer_call_fn_202130
*__inference_dropout_2_layer_call_fn_202125і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_dense_4_layer_call_and_return_conditional_losses_202141Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_4_layer_call_fn_202150Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202186
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202206і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
6__inference_batch_normalization_3_layer_call_fn_202232
6__inference_batch_normalization_3_layer_call_fn_202219і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_3_layer_call_and_return_conditional_losses_202249
E__inference_dropout_3_layer_call_and_return_conditional_losses_202244і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_3_layer_call_fn_202259
*__inference_dropout_3_layer_call_fn_202254і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_dense_5_layer_call_and_return_conditional_losses_202270Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_5_layer_call_fn_202279Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЋB»
$__inference_signature_wrapper_201771input_2"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Ы
!__inference__wrapped_model_201048v!"+(*)451Ґ.
'Ґ$
"К
input_2€€€€€€€€€Р
™ "1™.
,
dense_5!К
dense_5€€€€€€€€€Ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202057b3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ј
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202077b3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ П
6__inference_batch_normalization_2_layer_call_fn_202090U3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@П
6__inference_batch_normalization_2_layer_call_fn_202103U3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Ј
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202186b*+()3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Ј
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202206b+(*)3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ П
6__inference_batch_normalization_3_layer_call_fn_202219U*+()3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p
™ "К€€€€€€€€€$П
6__inference_batch_normalization_3_layer_call_fn_202232U+(*)3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p 
™ "К€€€€€€€€€$§
C__inference_dense_3_layer_call_and_return_conditional_losses_202012]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
(__inference_dense_3_layer_call_fn_202021P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€@£
C__inference_dense_4_layer_call_and_return_conditional_losses_202141\!"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€$
Ъ {
(__inference_dense_4_layer_call_fn_202150O!"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€$£
C__inference_dense_5_layer_call_and_return_conditional_losses_202270\45/Ґ,
%Ґ"
 К
inputs€€€€€€€€€$
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_5_layer_call_fn_202279O45/Ґ,
%Ґ"
 К
inputs€€€€€€€€€$
™ "К€€€€€€€€€•
E__inference_dropout_2_layer_call_and_return_conditional_losses_202115\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ •
E__inference_dropout_2_layer_call_and_return_conditional_losses_202120\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
*__inference_dropout_2_layer_call_fn_202125O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@}
*__inference_dropout_2_layer_call_fn_202130O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@•
E__inference_dropout_3_layer_call_and_return_conditional_losses_202244\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p
™ "%Ґ"
К
0€€€€€€€€€$
Ъ •
E__inference_dropout_3_layer_call_and_return_conditional_losses_202249\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ }
*__inference_dropout_3_layer_call_fn_202254O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p
™ "К€€€€€€€€€$}
*__inference_dropout_3_layer_call_fn_202259O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€$
p 
™ "К€€€€€€€€€$Њ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201544r!"*+()459Ґ6
/Ґ,
"К
input_2€€€€€€€€€Р
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Њ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201583r!"+(*)459Ґ6
/Ґ,
"К
input_2€€€€€€€€€Р
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201876q!"*+()458Ґ5
.Ґ+
!К
inputs€€€€€€€€€Р
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ љ
H__inference_sequential_1_layer_call_and_return_conditional_losses_201935q!"+(*)458Ґ5
.Ґ+
!К
inputs€€€€€€€€€Р
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ц
-__inference_sequential_1_layer_call_fn_201656e!"*+()459Ґ6
/Ґ,
"К
input_2€€€€€€€€€Р
p

 
™ "К€€€€€€€€€Ц
-__inference_sequential_1_layer_call_fn_201728e!"+(*)459Ґ6
/Ґ,
"К
input_2€€€€€€€€€Р
p 

 
™ "К€€€€€€€€€Х
-__inference_sequential_1_layer_call_fn_201968d!"*+()458Ґ5
.Ґ+
!К
inputs€€€€€€€€€Р
p

 
™ "К€€€€€€€€€Х
-__inference_sequential_1_layer_call_fn_202001d!"+(*)458Ґ5
.Ґ+
!К
inputs€€€€€€€€€Р
p 

 
™ "К€€€€€€€€€™
$__inference_signature_wrapper_201771Б!"+(*)45<Ґ9
Ґ 
2™/
-
input_2"К
input_2€€€€€€€€€Р"1™.
,
dense_5!К
dense_5€€€€€€€€€