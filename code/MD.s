	.file	"MD.cpp"
	.text
	.p2align 4
	.globl	_Z19MeanSquaredVelocityv
	.type	_Z19MeanSquaredVelocityv, @function
_Z19MeanSquaredVelocityv:
.LFB5585:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %edi
	leal	(%rdi,%rdi,2), %ecx
	testl	%ecx, %ecx
	jle	.L7
	leal	-1(%rcx), %eax
	cmpl	$2, %eax
	jbe	.L8
	movl	%ecx, %edx
	leaq	v(%rip), %rsi
	vxorpd	%xmm0, %xmm0, %xmm0
	shrl	$2, %edx
	movq	%rsi, %r9
	salq	$5, %rdx
	leaq	(%rsi,%rdx), %r8
	subq	$32, %rdx
	shrq	$5, %rdx
	addq	$1, %rdx
	andl	$7, %edx
	je	.L5
	cmpq	$1, %rdx
	je	.L30
	cmpq	$2, %rdx
	je	.L31
	cmpq	$3, %rdx
	je	.L32
	cmpq	$4, %rdx
	je	.L33
	cmpq	$5, %rdx
	je	.L34
	cmpq	$6, %rdx
	jne	.L46
.L35:
	vmovapd	(%r9), %ymm5
	addq	$32, %r9
	vmulpd	%ymm5, %ymm5, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
.L34:
	vmovapd	(%r9), %ymm4
	addq	$32, %r9
	vmulpd	%ymm4, %ymm4, %ymm2
	vaddpd	%ymm2, %ymm0, %ymm0
.L33:
	vmovapd	(%r9), %ymm3
	addq	$32, %r9
	vmulpd	%ymm3, %ymm3, %ymm7
	vaddpd	%ymm7, %ymm0, %ymm0
.L32:
	vmovapd	(%r9), %ymm8
	addq	$32, %r9
	vmulpd	%ymm8, %ymm8, %ymm9
	vaddpd	%ymm9, %ymm0, %ymm0
.L31:
	vmovapd	(%r9), %ymm10
	addq	$32, %r9
	vmulpd	%ymm10, %ymm10, %ymm11
	vaddpd	%ymm11, %ymm0, %ymm0
.L30:
	vmovapd	(%r9), %ymm12
	addq	$32, %r9
	vmulpd	%ymm12, %ymm12, %ymm13
	vaddpd	%ymm13, %ymm0, %ymm0
	cmpq	%r8, %r9
	je	.L9
.L5:
	vmovapd	(%r9), %ymm14
	vmovapd	32(%r9), %ymm5
	addq	$256, %r9
	vmovapd	-192(%r9), %ymm2
	vmovapd	-128(%r9), %ymm11
	vmovapd	-160(%r9), %ymm7
	vmulpd	%ymm14, %ymm14, %ymm15
	vmulpd	%ymm5, %ymm5, %ymm1
	vmulpd	%ymm2, %ymm2, %ymm3
	vmulpd	%ymm7, %ymm7, %ymm9
	vmulpd	%ymm11, %ymm11, %ymm12
	vaddpd	%ymm15, %ymm0, %ymm6
	vmovapd	-96(%r9), %ymm0
	vmulpd	%ymm0, %ymm0, %ymm14
	vaddpd	%ymm1, %ymm6, %ymm4
	vmovapd	-64(%r9), %ymm6
	vmovapd	-32(%r9), %ymm1
	vmulpd	%ymm6, %ymm6, %ymm5
	vaddpd	%ymm3, %ymm4, %ymm8
	vmulpd	%ymm1, %ymm1, %ymm2
	vaddpd	%ymm9, %ymm8, %ymm10
	vaddpd	%ymm12, %ymm10, %ymm13
	vaddpd	%ymm14, %ymm13, %ymm15
	vaddpd	%ymm5, %ymm15, %ymm4
	vaddpd	%ymm2, %ymm4, %ymm0
	cmpq	%r8, %r9
	jne	.L5
.L9:
	vextractf128	$0x1, %ymm0, %xmm3
	movl	%ecx, %eax
	vaddpd	%xmm0, %xmm3, %xmm8
	andl	$-4, %eax
	vunpckhpd	%xmm8, %xmm8, %xmm7
	vaddpd	%xmm8, %xmm7, %xmm0
	testb	$3, %cl
	je	.L47
	vzeroupper
.L3:
	movslq	%eax, %r10
	leal	1(%rax), %r11d
	vmovsd	(%rsi,%r10,8), %xmm9
	vmulsd	%xmm9, %xmm9, %xmm10
	vaddsd	%xmm10, %xmm0, %xmm0
	cmpl	%r11d, %ecx
	jle	.L2
	movslq	%r11d, %rdx
	addl	$2, %eax
	vmovsd	(%rsi,%rdx,8), %xmm11
	vmulsd	%xmm11, %xmm11, %xmm12
	vaddsd	%xmm12, %xmm0, %xmm0
	cmpl	%eax, %ecx
	jle	.L2
	cltq
	vmovsd	(%rsi,%rax,8), %xmm13
	vmulsd	%xmm13, %xmm13, %xmm14
	vaddsd	%xmm14, %xmm0, %xmm0
.L2:
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vxorps	%xmm15, %xmm15, %xmm15
	vcvtsi2sdl	%edi, %xmm15, %xmm6
	vdivsd	%xmm6, %xmm0, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L47:
	.cfi_restore_state
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vxorps	%xmm15, %xmm15, %xmm15
	vcvtsi2sdl	%edi, %xmm15, %xmm6
	vdivsd	%xmm6, %xmm0, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L46:
	.cfi_restore_state
	vmovapd	(%rsi), %ymm6
	leaq	32+v(%rip), %r9
	vmulpd	%ymm6, %ymm6, %ymm0
	jmp	.L35
	.p2align 4,,10
	.p2align 3
.L7:
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vxorps	%xmm15, %xmm15, %xmm15
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%edi, %xmm15, %xmm6
	vdivsd	%xmm6, %xmm0, %xmm0
	ret
.L8:
	.cfi_restore_state
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	leaq	v(%rip), %rsi
	jmp	.L3
	.cfi_endproc
.LFE5585:
	.size	_Z19MeanSquaredVelocityv, .-_Z19MeanSquaredVelocityv
	.p2align 4
	.globl	_Z7Kineticv
	.type	_Z7Kineticv, @function
_Z7Kineticv:
.LFB5586:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	leal	(%rax,%rax,2), %edi
	testl	%edi, %edi
	jle	.L54
	leal	-1(%rdi), %edx
	movl	$2863311531, %ecx
	vmovsd	.LC1(%rip), %xmm3
	vmulsd	m(%rip), %xmm3, %xmm10
	movq	%rdx, %rsi
	imulq	%rcx, %rdx
	shrq	$33, %rdx
	addl	$1, %edx
	cmpl	$8, %esi
	jbe	.L55
	movl	%edx, %r8d
	leaq	v(%rip), %r9
	vmovddup	%xmm10, %xmm4
	shrl	$2, %r8d
	vinsertf128	$1, %xmm4, %ymm4, %ymm4
	vxorpd	%xmm9, %xmm9, %xmm9
	movq	%r9, %rax
	leaq	(%r8,%r8,2), %r10
	salq	$5, %r10
	leaq	(%r9,%r10), %r11
	andl	$32, %r10d
	je	.L52
	vmovapd	(%r9), %ymm5
	vmovapd	64+v(%rip), %ymm2
	leaq	96+v(%rip), %rax
	vperm2f128	$48, 32+v(%rip), %ymm5, %ymm0
	vperm2f128	$2, 32+v(%rip), %ymm5, %ymm9
	vperm2f128	$33, %ymm5, %ymm5, %ymm6
	vpermilpd	$2, %ymm2, %ymm12
	vshufpd	$2, %ymm6, %ymm0, %ymm1
	vperm2f128	$33, %ymm2, %ymm1, %ymm7
	vshufpd	$5, %ymm9, %ymm0, %ymm11
	vshufpd	$2, %ymm9, %ymm6, %ymm14
	vblendpd	$8, %ymm7, %ymm1, %ymm8
	vinsertf128	$1, %xmm2, %ymm14, %ymm15
	vblendpd	$8, %ymm12, %ymm11, %ymm13
	vmulpd	%ymm8, %ymm8, %ymm5
	vblendpd	$7, %ymm15, %ymm2, %ymm3
	vmulpd	%ymm13, %ymm13, %ymm2
	vmulpd	%ymm3, %ymm3, %ymm6
	vaddpd	%ymm2, %ymm5, %ymm0
	vaddpd	%ymm6, %ymm0, %ymm1
	vmulpd	%ymm4, %ymm1, %ymm9
	cmpq	%r11, %rax
	je	.L51
	.p2align 4,,10
	.p2align 3
.L52:
	vmovapd	(%rax), %ymm11
	vmovapd	64(%rax), %ymm12
	addq	$192, %rax
	vperm2f128	$48, -160(%rax), %ymm11, %ymm13
	vperm2f128	$2, -160(%rax), %ymm11, %ymm3
	vperm2f128	$33, %ymm11, %ymm11, %ymm7
	vpermilpd	$2, %ymm12, %ymm2
	vshufpd	$2, %ymm7, %ymm13, %ymm14
	vperm2f128	$33, %ymm12, %ymm14, %ymm8
	vshufpd	$5, %ymm3, %ymm13, %ymm5
	vshufpd	$2, %ymm3, %ymm7, %ymm6
	vblendpd	$8, %ymm8, %ymm14, %ymm15
	vblendpd	$8, %ymm2, %ymm5, %ymm0
	vinsertf128	$1, %xmm12, %ymm6, %ymm1
	vmovapd	-96(%rax), %ymm2
	vblendpd	$7, %ymm1, %ymm12, %ymm11
	vmovapd	-64(%rax), %ymm6
	vmovapd	-32(%rax), %ymm5
	vmulpd	%ymm15, %ymm15, %ymm12
	vmulpd	%ymm0, %ymm0, %ymm13
	vperm2f128	$48, %ymm6, %ymm2, %ymm0
	vmulpd	%ymm11, %ymm11, %ymm14
	vperm2f128	$33, %ymm2, %ymm2, %ymm11
	vshufpd	$2, %ymm11, %ymm0, %ymm1
	vaddpd	%ymm13, %ymm12, %ymm7
	vperm2f128	$2, %ymm6, %ymm2, %ymm13
	vaddpd	%ymm14, %ymm7, %ymm8
	vpermilpd	$2, %ymm5, %ymm14
	vshufpd	$5, %ymm13, %ymm0, %ymm7
	vmulpd	%ymm4, %ymm8, %ymm15
	vblendpd	$8, %ymm14, %ymm7, %ymm8
	vmulpd	%ymm8, %ymm8, %ymm0
	vaddpd	%ymm15, %ymm9, %ymm3
	vperm2f128	$33, %ymm5, %ymm1, %ymm9
	vshufpd	$2, %ymm13, %ymm11, %ymm15
	vinsertf128	$1, %xmm5, %ymm15, %ymm2
	vblendpd	$8, %ymm9, %ymm1, %ymm12
	vblendpd	$7, %ymm2, %ymm5, %ymm6
	vmulpd	%ymm12, %ymm12, %ymm5
	vmulpd	%ymm6, %ymm6, %ymm1
	vaddpd	%ymm0, %ymm5, %ymm11
	vaddpd	%ymm1, %ymm11, %ymm9
	vmulpd	%ymm4, %ymm9, %ymm12
	vaddpd	%ymm12, %ymm3, %ymm9
	cmpq	%r11, %rax
	jne	.L52
.L51:
	vextractf128	$0x1, %ymm9, %xmm4
	movl	%edx, %esi
	vaddpd	%xmm9, %xmm4, %xmm3
	andl	$-4, %esi
	leal	(%rsi,%rsi,2), %eax
	vunpckhpd	%xmm3, %xmm3, %xmm13
	vaddpd	%xmm3, %xmm13, %xmm0
	cmpl	%edx, %esi
	je	.L63
	vzeroupper
.L50:
	leal	1(%rax), %ecx
	leal	2(%rax), %r10d
	movslq	%eax, %rdx
	movslq	%ecx, %r8
	movslq	%r10d, %r11
	vmovsd	(%r9,%rdx,8), %xmm7
	leal	3(%rax), %esi
	vmovsd	(%r9,%r8,8), %xmm14
	vmovsd	(%r9,%r11,8), %xmm8
	vmulsd	%xmm7, %xmm7, %xmm5
	vmulsd	%xmm14, %xmm14, %xmm15
	vmulsd	%xmm8, %xmm8, %xmm2
	vaddsd	%xmm2, %xmm15, %xmm6
	vaddsd	%xmm5, %xmm6, %xmm11
	vmulsd	%xmm10, %xmm11, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpl	%esi, %edi
	jle	.L62
	leal	4(%rax), %ecx
	leal	5(%rax), %r10d
	movslq	%esi, %rdx
	movslq	%ecx, %r8
	movslq	%r10d, %r11
	vmovsd	(%r9,%rdx,8), %xmm9
	leal	6(%rax), %esi
	vmovsd	(%r9,%r8,8), %xmm12
	vmovsd	(%r9,%r11,8), %xmm4
	vmulsd	%xmm9, %xmm9, %xmm14
	vmulsd	%xmm12, %xmm12, %xmm3
	vmulsd	%xmm4, %xmm4, %xmm13
	vaddsd	%xmm13, %xmm3, %xmm7
	vaddsd	%xmm14, %xmm7, %xmm8
	vmulsd	%xmm10, %xmm8, %xmm15
	vaddsd	%xmm15, %xmm0, %xmm0
	cmpl	%esi, %edi
	jle	.L62
	leal	7(%rax), %edx
	addl	$8, %eax
	movslq	%esi, %rdi
	movslq	%edx, %rcx
	cltq
	vmovsd	(%r9,%rdi,8), %xmm2
	vmovsd	(%r9,%rcx,8), %xmm6
	vmovsd	(%r9,%rax,8), %xmm5
	vmulsd	%xmm2, %xmm2, %xmm12
	vmulsd	%xmm5, %xmm5, %xmm11
	vmulsd	%xmm6, %xmm6, %xmm1
	vaddsd	%xmm1, %xmm11, %xmm9
	vaddsd	%xmm12, %xmm9, %xmm4
	vmulsd	%xmm10, %xmm4, %xmm10
	vaddsd	%xmm10, %xmm0, %xmm0
.L62:
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L63:
	.cfi_restore_state
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L54:
	.cfi_restore_state
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vxorpd	%xmm0, %xmm0, %xmm0
	ret
.L55:
	.cfi_restore_state
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	leaq	v(%rip), %r9
	jmp	.L50
	.cfi_endproc
.LFE5586:
	.size	_Z7Kineticv, .-_Z7Kineticv
	.p2align 4
	.globl	_Z12PotentialAuxddd
	.type	_Z12PotentialAuxddd, @function
_Z12PotentialAuxddd:
.LFB5587:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
1:	call	*mcount@GOTPCREL(%rip)
	vmovsd	sigma(%rip), %xmm5
	popq	%rbp
	.cfi_def_cfa 7, 8
	vmulsd	%xmm1, %xmm1, %xmm1
	vmulsd	%xmm2, %xmm2, %xmm2
	vmulsd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm2, %xmm1, %xmm3
	vaddsd	%xmm0, %xmm3, %xmm4
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm4, %xmm5, %xmm6
	vmulsd	%xmm6, %xmm6, %xmm7
	vmulsd	%xmm6, %xmm7, %xmm8
	vmulsd	%xmm8, %xmm8, %xmm9
	vsubsd	.LC2(%rip), %xmm9, %xmm10
	vmulsd	%xmm9, %xmm10, %xmm0
	ret
	.cfi_endproc
.LFE5587:
	.size	_Z12PotentialAuxddd, .-_Z12PotentialAuxddd
	.p2align 4
	.globl	_Z9Potentialv
	.type	_Z9Potentialv, @function
_Z9Potentialv:
.LFB5588:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	leal	(%rax,%rax,2), %r9d
	testl	%r9d, %r9d
	jle	.L77
	vmovsd	sigma(%rip), %xmm5
	vmovsd	%xmm5, 24(%rsp)
	cmpl	$3, %r9d
	jle	.L77
	leaq	24+r(%rip), %rcx
	vbroadcastsd	24(%rsp), %ymm6
	leal	-4(%r9), %r10d
	vxorpd	%xmm5, %xmm5, %xmm5
	movl	$3, %r8d
	leaq	-24(%rcx), %rdx
	movl	$1, %ebx
	vmovapd	.LC3(%rip), %ymm9
	movl	$2863311531, %r11d
	.p2align 4,,10
	.p2align 3
.L71:
	movl	%r10d, %esi
	vmovsd	-8(%rcx), %xmm4
	vmovsd	-16(%rcx), %xmm7
	imulq	%r11, %rsi
	vmovsd	-24(%rcx), %xmm8
	shrq	$33, %rsi
	addl	$1, %esi
	cmpl	%r8d, %r9d
	cmovle	%ebx, %esi
	cmpl	$8, %r10d
	jbe	.L79
	cmpl	%r8d, %r9d
	jle	.L79
	movl	%esi, %edi
	vmovddup	%xmm4, %xmm12
	vmovddup	%xmm7, %xmm11
	movq	%rcx, %rax
	shrl	$2, %edi
	vmovddup	%xmm8, %xmm10
	vinsertf128	$1, %xmm12, %ymm12, %ymm12
	leaq	(%rdi,%rdi,2), %rdi
	vinsertf128	$1, %xmm11, %ymm11, %ymm11
	vinsertf128	$1, %xmm10, %ymm10, %ymm10
	salq	$5, %rdi
	vxorpd	%xmm13, %xmm13, %xmm13
	leaq	(%rdi,%rcx), %r12
	andl	$32, %edi
	je	.L73
	vmovupd	(%rcx), %xmm3
	vmovupd	32(%rcx), %xmm2
	leaq	96(%rcx), %rax
	vinsertf128	$0x1, 48(%rcx), %ymm2, %ymm14
	vinsertf128	$0x1, 16(%rcx), %ymm3, %ymm0
	vmovupd	64(%rcx), %xmm1
	vinsertf128	$0x1, 80(%rcx), %ymm1, %ymm15
	vperm2f128	$2, %ymm14, %ymm0, %ymm3
	vperm2f128	$33, %ymm0, %ymm0, %ymm13
	vperm2f128	$48, %ymm14, %ymm0, %ymm0
	vshufpd	$2, %ymm3, %ymm13, %ymm2
	vinsertf128	$1, %xmm15, %ymm2, %ymm1
	vpermilpd	$2, %ymm15, %ymm14
	vshufpd	$2, %ymm13, %ymm0, %ymm13
	vblendpd	$7, %ymm1, %ymm15, %ymm2
	vperm2f128	$33, %ymm15, %ymm13, %ymm15
	vshufpd	$5, %ymm3, %ymm0, %ymm3
	vsubpd	%ymm2, %ymm12, %ymm1
	vblendpd	$8, %ymm15, %ymm13, %ymm0
	vblendpd	$8, %ymm14, %ymm3, %ymm2
	vsubpd	%ymm2, %ymm11, %ymm14
	vsubpd	%ymm0, %ymm10, %ymm3
	vmulpd	%ymm1, %ymm1, %ymm1
	vmulpd	%ymm3, %ymm3, %ymm13
	vmulpd	%ymm14, %ymm14, %ymm2
	vaddpd	%ymm2, %ymm13, %ymm14
	vaddpd	%ymm1, %ymm14, %ymm15
	vsqrtpd	%ymm15, %ymm0
	vdivpd	%ymm0, %ymm6, %ymm3
	vmulpd	%ymm3, %ymm3, %ymm13
	vmulpd	%ymm3, %ymm13, %ymm2
	vmulpd	%ymm2, %ymm2, %ymm14
	vaddpd	%ymm9, %ymm14, %ymm1
	vmulpd	%ymm14, %ymm1, %ymm13
	cmpq	%r12, %rax
	je	.L98
	.p2align 4,,10
	.p2align 3
.L73:
	vmovupd	(%rax), %xmm15
	vmovupd	32(%rax), %xmm0
	addq	$192, %rax
	vinsertf128	$0x1, -144(%rax), %ymm0, %ymm1
	vmovupd	-128(%rax), %xmm3
	vinsertf128	$0x1, -176(%rax), %ymm15, %ymm2
	vinsertf128	$0x1, -112(%rax), %ymm3, %ymm3
	vperm2f128	$2, %ymm1, %ymm2, %ymm14
	vperm2f128	$33, %ymm2, %ymm2, %ymm15
	vperm2f128	$48, %ymm1, %ymm2, %ymm2
	vshufpd	$2, %ymm14, %ymm15, %ymm0
	vinsertf128	$1, %xmm3, %ymm0, %ymm0
	vshufpd	$5, %ymm14, %ymm2, %ymm1
	vshufpd	$2, %ymm15, %ymm2, %ymm15
	vpermilpd	$2, %ymm3, %ymm14
	vblendpd	$7, %ymm0, %ymm3, %ymm0
	vperm2f128	$33, %ymm3, %ymm15, %ymm3
	vblendpd	$8, %ymm3, %ymm15, %ymm2
	vblendpd	$8, %ymm14, %ymm1, %ymm1
	vsubpd	%ymm0, %ymm12, %ymm0
	vsubpd	%ymm1, %ymm11, %ymm14
	vsubpd	%ymm2, %ymm10, %ymm1
	vmulpd	%ymm0, %ymm0, %ymm0
	vmulpd	%ymm1, %ymm1, %ymm15
	vmulpd	%ymm14, %ymm14, %ymm14
	vaddpd	%ymm14, %ymm15, %ymm3
	vaddpd	%ymm0, %ymm3, %ymm2
	vsqrtpd	%ymm2, %ymm1
	vdivpd	%ymm1, %ymm6, %ymm15
	vmulpd	%ymm15, %ymm15, %ymm14
	vmulpd	%ymm15, %ymm14, %ymm3
	vmovupd	-96(%rax), %xmm15
	vmovupd	-64(%rax), %xmm14
	vmulpd	%ymm3, %ymm3, %ymm2
	vmovupd	-32(%rax), %xmm3
	vinsertf128	$0x1, -16(%rax), %ymm3, %ymm3
	vaddpd	%ymm9, %ymm2, %ymm0
	vmulpd	%ymm2, %ymm0, %ymm1
	vinsertf128	$0x1, -48(%rax), %ymm14, %ymm0
	vaddpd	%ymm1, %ymm13, %ymm13
	vinsertf128	$0x1, -80(%rax), %ymm15, %ymm1
	vperm2f128	$2, %ymm0, %ymm1, %ymm14
	vperm2f128	$33, %ymm1, %ymm1, %ymm15
	vperm2f128	$48, %ymm0, %ymm1, %ymm1
	vshufpd	$2, %ymm14, %ymm15, %ymm2
	vinsertf128	$1, %xmm3, %ymm2, %ymm2
	vshufpd	$5, %ymm14, %ymm1, %ymm0
	vshufpd	$2, %ymm15, %ymm1, %ymm15
	vpermilpd	$2, %ymm3, %ymm14
	vblendpd	$7, %ymm2, %ymm3, %ymm2
	vperm2f128	$33, %ymm3, %ymm15, %ymm3
	vblendpd	$8, %ymm3, %ymm15, %ymm1
	vblendpd	$8, %ymm14, %ymm0, %ymm0
	vsubpd	%ymm2, %ymm12, %ymm2
	vsubpd	%ymm0, %ymm11, %ymm14
	vsubpd	%ymm1, %ymm10, %ymm0
	vmulpd	%ymm2, %ymm2, %ymm2
	vmulpd	%ymm0, %ymm0, %ymm15
	vmulpd	%ymm14, %ymm14, %ymm14
	vaddpd	%ymm14, %ymm15, %ymm3
	vaddpd	%ymm2, %ymm3, %ymm1
	vsqrtpd	%ymm1, %ymm0
	vdivpd	%ymm0, %ymm6, %ymm15
	vmulpd	%ymm15, %ymm15, %ymm14
	vmulpd	%ymm15, %ymm14, %ymm3
	vmulpd	%ymm3, %ymm3, %ymm2
	vaddpd	%ymm9, %ymm2, %ymm1
	vmulpd	%ymm2, %ymm1, %ymm0
	vaddpd	%ymm0, %ymm13, %ymm13
	cmpq	%r12, %rax
	jne	.L73
.L98:
	vextractf128	$0x1, %ymm13, %xmm12
	movl	%esi, %edi
	vaddpd	%xmm13, %xmm12, %xmm11
	andl	$-4, %edi
	leal	(%rdi,%rdi,2), %eax
	addl	%r8d, %eax
	vunpckhpd	%xmm11, %xmm11, %xmm10
	vaddpd	%xmm11, %xmm10, %xmm13
	vaddsd	%xmm13, %xmm5, %xmm5
	cmpl	%esi, %edi
	je	.L74
.L72:
	leal	2(%rax), %esi
	leal	1(%rax), %edi
	vmovsd	24(%rsp), %xmm10
	movslq	%esi, %r12
	movslq	%edi, %rsi
	vsubsd	(%rdx,%r12,8), %xmm4, %xmm15
	vsubsd	(%rdx,%rsi,8), %xmm7, %xmm14
	movslq	%eax, %r12
	vsubsd	(%rdx,%r12,8), %xmm8, %xmm3
	leal	3(%rax), %esi
	vmulsd	%xmm14, %xmm14, %xmm1
	vmulsd	%xmm15, %xmm15, %xmm2
	vmulsd	%xmm3, %xmm3, %xmm12
	vaddsd	%xmm2, %xmm1, %xmm0
	vaddsd	%xmm12, %xmm0, %xmm11
	vsqrtsd	%xmm11, %xmm11, %xmm11
	vdivsd	%xmm11, %xmm10, %xmm13
	vmulsd	%xmm13, %xmm13, %xmm15
	vmulsd	%xmm13, %xmm15, %xmm14
	vmulsd	%xmm14, %xmm14, %xmm3
	vsubsd	.LC2(%rip), %xmm3, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm5, %xmm5
	cmpl	%esi, %r9d
	jle	.L74
	leal	5(%rax), %edi
	movslq	%esi, %rsi
	movslq	%edi, %r12
	leal	4(%rax), %edi
	vsubsd	(%rdx,%rsi,8), %xmm8, %xmm11
	vsubsd	(%rdx,%r12,8), %xmm4, %xmm12
	movslq	%edi, %r12
	vsubsd	(%rdx,%r12,8), %xmm7, %xmm0
	leal	6(%rax), %r12d
	vmulsd	%xmm11, %xmm11, %xmm3
	vmulsd	%xmm12, %xmm12, %xmm15
	vmulsd	%xmm0, %xmm0, %xmm13
	vaddsd	%xmm15, %xmm13, %xmm14
	vaddsd	%xmm3, %xmm14, %xmm1
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm10, %xmm2
	vmulsd	%xmm2, %xmm2, %xmm12
	vmulsd	%xmm2, %xmm12, %xmm0
	vmulsd	%xmm0, %xmm0, %xmm11
	vsubsd	.LC2(%rip), %xmm11, %xmm13
	vmulsd	%xmm11, %xmm13, %xmm15
	vaddsd	%xmm15, %xmm5, %xmm5
	cmpl	%r12d, %r9d
	jle	.L74
	leal	8(%rax), %edi
	addl	$7, %eax
	cltq
	movslq	%edi, %rsi
	vsubsd	(%rdx,%rsi,8), %xmm4, %xmm4
	vsubsd	(%rdx,%rax,8), %xmm7, %xmm7
	movslq	%r12d, %rax
	vsubsd	(%rdx,%rax,8), %xmm8, %xmm8
	vmulsd	%xmm4, %xmm4, %xmm14
	vmulsd	%xmm7, %xmm7, %xmm3
	vmulsd	%xmm8, %xmm8, %xmm2
	vaddsd	%xmm3, %xmm14, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm12
	vsqrtsd	%xmm12, %xmm12, %xmm12
	vdivsd	%xmm12, %xmm10, %xmm10
	vmulsd	%xmm10, %xmm10, %xmm0
	vmulsd	%xmm10, %xmm0, %xmm11
	vmulsd	%xmm11, %xmm11, %xmm13
	vsubsd	.LC2(%rip), %xmm13, %xmm15
	vmulsd	%xmm13, %xmm15, %xmm4
	vaddsd	%xmm4, %xmm5, %xmm5
.L74:
	leal	-1(%r8), %esi
	vmovsd	16(%rcx), %xmm4
	vmovsd	8(%rcx), %xmm7
	movq	%rsi, %r12
	imulq	%r11, %rsi
	vmovsd	(%rcx), %xmm8
	shrq	$33, %rsi
	addl	$1, %esi
	cmpl	$8, %r12d
	jbe	.L102
	vmovddup	%xmm4, %xmm12
	vmovddup	%xmm7, %xmm11
	vmovddup	%xmm8, %xmm10
	movl	%esi, %edi
	shrl	$2, %edi
	vinsertf128	$1, %xmm12, %ymm12, %ymm12
	vxorpd	%xmm13, %xmm13, %xmm13
	leaq	(%rdi,%rdi,2), %rdi
	leaq	r(%rip), %rax
	vinsertf128	$1, %xmm11, %ymm11, %ymm11
	salq	$5, %rdi
	vinsertf128	$1, %xmm10, %ymm10, %ymm10
	addq	%rdx, %rdi
	movq	%rdi, %r12
	subq	%rax, %r12
	andl	$32, %r12d
	je	.L69
	vmovapd	r(%rip), %ymm1
	vmovapd	64+r(%rip), %ymm3
	leaq	96+r(%rip), %rax
	vperm2f128	$2, 32+r(%rip), %ymm1, %ymm0
	vperm2f128	$33, %ymm1, %ymm1, %ymm14
	vperm2f128	$48, 32+r(%rip), %ymm1, %ymm1
	vshufpd	$2, %ymm0, %ymm14, %ymm2
	vinsertf128	$1, %xmm3, %ymm2, %ymm13
	vblendpd	$7, %ymm13, %ymm3, %ymm15
	vshufpd	$2, %ymm14, %ymm1, %ymm14
	vshufpd	$5, %ymm0, %ymm1, %ymm0
	vpermilpd	$2, %ymm3, %ymm13
	vperm2f128	$33, %ymm3, %ymm14, %ymm3
	vsubpd	%ymm15, %ymm12, %ymm2
	vblendpd	$8, %ymm13, %ymm0, %ymm15
	vblendpd	$8, %ymm3, %ymm14, %ymm1
	vsubpd	%ymm15, %ymm11, %ymm0
	vsubpd	%ymm1, %ymm10, %ymm13
	vmulpd	%ymm2, %ymm2, %ymm2
	vmulpd	%ymm0, %ymm0, %ymm15
	vmulpd	%ymm13, %ymm13, %ymm0
	vaddpd	%ymm0, %ymm15, %ymm14
	vaddpd	%ymm2, %ymm14, %ymm3
	vsqrtpd	%ymm3, %ymm1
	vdivpd	%ymm1, %ymm6, %ymm15
	vmulpd	%ymm15, %ymm15, %ymm13
	vmulpd	%ymm15, %ymm13, %ymm0
	vmulpd	%ymm0, %ymm0, %ymm14
	vaddpd	.LC3(%rip), %ymm14, %ymm2
	vmulpd	%ymm14, %ymm2, %ymm13
	cmpq	%rdi, %rax
	je	.L97
	.p2align 4,,10
	.p2align 3
.L69:
	vmovapd	(%rax), %ymm1
	vmovapd	64(%rax), %ymm14
	addq	$192, %rax
	vperm2f128	$2, -160(%rax), %ymm1, %ymm2
	vperm2f128	$33, %ymm1, %ymm1, %ymm15
	vshufpd	$2, %ymm2, %ymm15, %ymm3
	vinsertf128	$1, %xmm14, %ymm3, %ymm0
	vblendpd	$7, %ymm0, %ymm14, %ymm3
	vperm2f128	$48, -160(%rax), %ymm1, %ymm0
	vpermilpd	$2, %ymm14, %ymm1
	vsubpd	%ymm3, %ymm12, %ymm3
	vshufpd	$2, %ymm15, %ymm0, %ymm15
	vperm2f128	$33, %ymm14, %ymm15, %ymm14
	vshufpd	$5, %ymm2, %ymm0, %ymm2
	vblendpd	$8, %ymm1, %ymm2, %ymm2
	vblendpd	$8, %ymm14, %ymm15, %ymm0
	vsubpd	%ymm2, %ymm11, %ymm2
	vsubpd	%ymm0, %ymm10, %ymm1
	vmulpd	%ymm3, %ymm3, %ymm3
	vmulpd	%ymm2, %ymm2, %ymm2
	vmulpd	%ymm1, %ymm1, %ymm15
	vaddpd	%ymm15, %ymm2, %ymm14
	vaddpd	%ymm3, %ymm14, %ymm0
	vsqrtpd	%ymm0, %ymm1
	vdivpd	%ymm1, %ymm6, %ymm2
	vmulpd	%ymm2, %ymm2, %ymm15
	vmulpd	%ymm2, %ymm15, %ymm14
	vmovapd	-96(%rax), %ymm2
	vperm2f128	$33, %ymm2, %ymm2, %ymm15
	vmulpd	%ymm14, %ymm14, %ymm3
	vmovapd	-32(%rax), %ymm14
	vaddpd	.LC3(%rip), %ymm3, %ymm0
	vmulpd	%ymm3, %ymm0, %ymm1
	vmovapd	-64(%rax), %ymm0
	vaddpd	%ymm1, %ymm13, %ymm13
	vperm2f128	$2, %ymm0, %ymm2, %ymm1
	vperm2f128	$48, %ymm0, %ymm2, %ymm2
	vshufpd	$2, %ymm1, %ymm15, %ymm3
	vinsertf128	$1, %xmm14, %ymm3, %ymm3
	vpermilpd	$2, %ymm14, %ymm0
	vshufpd	$2, %ymm15, %ymm2, %ymm15
	vblendpd	$7, %ymm3, %ymm14, %ymm3
	vperm2f128	$33, %ymm14, %ymm15, %ymm14
	vshufpd	$5, %ymm1, %ymm2, %ymm1
	vblendpd	$8, %ymm0, %ymm1, %ymm1
	vblendpd	$8, %ymm14, %ymm15, %ymm2
	vsubpd	%ymm1, %ymm11, %ymm1
	vsubpd	%ymm3, %ymm12, %ymm3
	vsubpd	%ymm2, %ymm10, %ymm0
	vmulpd	%ymm1, %ymm1, %ymm1
	vmulpd	%ymm0, %ymm0, %ymm15
	vmulpd	%ymm3, %ymm3, %ymm3
	vaddpd	%ymm15, %ymm1, %ymm14
	vaddpd	%ymm3, %ymm14, %ymm2
	vsqrtpd	%ymm2, %ymm0
	vdivpd	%ymm0, %ymm6, %ymm15
	vmulpd	%ymm15, %ymm15, %ymm1
	vmulpd	%ymm15, %ymm1, %ymm14
	vmulpd	%ymm14, %ymm14, %ymm3
	vaddpd	.LC3(%rip), %ymm3, %ymm2
	vmulpd	%ymm3, %ymm2, %ymm0
	vaddpd	%ymm0, %ymm13, %ymm13
	cmpq	%rdi, %rax
	jne	.L69
.L97:
	vextractf128	$0x1, %ymm13, %xmm12
	movl	%esi, %edi
	vaddpd	%xmm13, %xmm12, %xmm11
	andl	$-4, %edi
	leal	(%rdi,%rdi,2), %eax
	vunpckhpd	%xmm11, %xmm11, %xmm10
	vaddpd	%xmm11, %xmm10, %xmm13
	vaddsd	%xmm13, %xmm5, %xmm5
	cmpl	%edi, %esi
	je	.L70
.L75:
	leal	2(%rax), %esi
	leal	1(%rax), %edi
	vmovsd	24(%rsp), %xmm10
	movslq	%esi, %r12
	movslq	%edi, %rsi
	leal	-3(%r8), %edi
	vsubsd	(%rdx,%r12,8), %xmm4, %xmm15
	vsubsd	(%rdx,%rsi,8), %xmm7, %xmm14
	movslq	%eax, %r12
	vsubsd	(%rdx,%r12,8), %xmm8, %xmm1
	leal	3(%rax), %r12d
	vmulsd	%xmm14, %xmm14, %xmm3
	vmulsd	%xmm15, %xmm15, %xmm2
	vmulsd	%xmm1, %xmm1, %xmm12
	vaddsd	%xmm2, %xmm3, %xmm0
	vaddsd	%xmm12, %xmm0, %xmm11
	vsqrtsd	%xmm11, %xmm11, %xmm11
	vdivsd	%xmm11, %xmm10, %xmm13
	vmulsd	%xmm13, %xmm13, %xmm15
	vmulsd	%xmm13, %xmm15, %xmm14
	vmulsd	%xmm14, %xmm14, %xmm3
	vsubsd	.LC2(%rip), %xmm3, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm5, %xmm5
	cmpl	%edi, %eax
	jge	.L70
	leal	5(%rax), %esi
	movslq	%esi, %rsi
	vsubsd	(%rdx,%rsi,8), %xmm4, %xmm12
	leal	4(%rax), %esi
	movslq	%esi, %rsi
	vsubsd	(%rdx,%rsi,8), %xmm7, %xmm0
	movslq	%r12d, %rsi
	vmulsd	%xmm12, %xmm12, %xmm15
	vsubsd	(%rdx,%rsi,8), %xmm8, %xmm11
	leal	6(%rax), %esi
	vmulsd	%xmm0, %xmm0, %xmm13
	vmulsd	%xmm11, %xmm11, %xmm3
	vaddsd	%xmm15, %xmm13, %xmm14
	vaddsd	%xmm3, %xmm14, %xmm1
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm10, %xmm2
	vmulsd	%xmm2, %xmm2, %xmm12
	vmulsd	%xmm2, %xmm12, %xmm0
	vmulsd	%xmm0, %xmm0, %xmm11
	vsubsd	.LC2(%rip), %xmm11, %xmm13
	vmulsd	%xmm11, %xmm13, %xmm15
	vaddsd	%xmm15, %xmm5, %xmm5
	cmpl	%edi, %r12d
	jge	.L70
	leal	8(%rax), %r12d
	addl	$7, %eax
	cltq
	movslq	%r12d, %rdi
	vsubsd	(%rdx,%rdi,8), %xmm4, %xmm4
	vsubsd	(%rdx,%rax,8), %xmm7, %xmm7
	movslq	%esi, %rax
	vsubsd	(%rdx,%rax,8), %xmm8, %xmm8
	vmulsd	%xmm4, %xmm4, %xmm14
	vmulsd	%xmm7, %xmm7, %xmm3
	vmulsd	%xmm8, %xmm8, %xmm2
	vaddsd	%xmm3, %xmm14, %xmm1
	vaddsd	%xmm2, %xmm1, %xmm12
	vsqrtsd	%xmm12, %xmm12, %xmm12
	vdivsd	%xmm12, %xmm10, %xmm10
	vmulsd	%xmm10, %xmm10, %xmm0
	vmulsd	%xmm10, %xmm0, %xmm11
	vmulsd	%xmm11, %xmm11, %xmm13
	vsubsd	.LC2(%rip), %xmm13, %xmm15
	vmulsd	%xmm13, %xmm15, %xmm4
	vaddsd	%xmm4, %xmm5, %xmm5
.L70:
	addl	$3, %r8d
	subl	$3, %r10d
	addq	$24, %rcx
	cmpl	%r8d, %r9d
	jg	.L71
	vzeroupper
.L66:
	vmovsd	.LC4(%rip), %xmm6
	vmulsd	epsilon(%rip), %xmm6, %xmm9
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vmulsd	%xmm5, %xmm9, %xmm0
	ret
.L102:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L75
.L79:
	movl	%r8d, %eax
	jmp	.L72
.L77:
	vxorpd	%xmm5, %xmm5, %xmm5
	jmp	.L66
	.cfi_endproc
.LFE5588:
	.size	_Z9Potentialv, .-_Z9Potentialv
	.p2align 4
	.globl	_Z20computeAccelerationsv
	.type	_Z20computeAccelerationsv, @function
_Z20computeAccelerationsv:
.LFB5589:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$8, %rsp
	.cfi_offset 3, -24
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	leal	(%rax,%rax,2), %ebx
	testl	%ebx, %ebx
	jle	.L107
	leal	-1(%rbx), %edx
	xorl	%esi, %esi
	leaq	a(%rip), %rdi
	leaq	8(,%rdx,8), %rdx
	call	memset@PLT
.L107:
	leal	-3(%rbx), %r9d
	testl	%r9d, %r9d
	jle	.L113
	vmovsd	.LC2(%rip), %xmm5
	vmovsd	.LC5(%rip), %xmm9
	xorl	%r8d, %r8d
	leaq	r(%rip), %rdi
	leaq	a(%rip), %r10
	.p2align 4,,10
	.p2align 3
.L109:
	addl	$3, %r8d
	cmpl	%r8d, %ebx
	jle	.L110
	vmovsd	(%rdi), %xmm8
	vmovsd	8(%rdi), %xmm7
	movq	%r10, %r11
	movq	%rdi, %rcx
	vmovsd	16(%rdi), %xmm6
	movl	%r8d, %esi
	.p2align 4,,10
	.p2align 3
.L111:
	vsubsd	24(%rcx), %xmm8, %xmm4
	addl	$3, %esi
	addq	$24, %rcx
	addq	$24, %r11
	vsubsd	8(%rcx), %xmm7, %xmm3
	vsubsd	16(%rcx), %xmm6, %xmm2
	vmulsd	%xmm4, %xmm4, %xmm0
	vmulsd	%xmm3, %xmm3, %xmm1
	vmulsd	%xmm2, %xmm2, %xmm11
	vaddsd	%xmm1, %xmm0, %xmm10
	vaddsd	%xmm11, %xmm10, %xmm12
	vdivsd	%xmm12, %xmm5, %xmm13
	vmulsd	%xmm13, %xmm13, %xmm14
	vmulsd	%xmm9, %xmm13, %xmm11
	vmulsd	%xmm13, %xmm14, %xmm15
	vmovsd	(%r11), %xmm13
	vaddsd	%xmm15, %xmm15, %xmm0
	vsubsd	%xmm5, %xmm0, %xmm1
	vmulsd	%xmm15, %xmm1, %xmm10
	vmovsd	8(%r11), %xmm15
	vmovsd	16(%r11), %xmm1
	vmulsd	%xmm11, %xmm10, %xmm12
	vmulsd	%xmm12, %xmm4, %xmm4
	vmulsd	%xmm12, %xmm2, %xmm2
	vmulsd	%xmm12, %xmm3, %xmm3
	vsubsd	%xmm4, %xmm13, %xmm14
	vsubsd	%xmm2, %xmm1, %xmm10
	vsubsd	%xmm3, %xmm15, %xmm0
	vmovsd	%xmm14, (%r11)
	vmovsd	%xmm10, 16(%r11)
	vmovsd	%xmm0, 8(%r11)
	vaddsd	(%r10), %xmm4, %xmm11
	vaddsd	8(%r10), %xmm3, %xmm12
	vaddsd	16(%r10), %xmm2, %xmm4
	vmovsd	%xmm11, (%r10)
	vmovsd	%xmm12, 8(%r10)
	vmovsd	%xmm4, 16(%r10)
	cmpl	%esi, %ebx
	jg	.L111
.L110:
	addq	$24, %rdi
	addq	$24, %r10
	cmpl	%r9d, %r8d
	jl	.L109
.L113:
	addq	$8, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5589:
	.size	_Z20computeAccelerationsv, .-_Z20computeAccelerationsv
	.p2align 4
	.globl	_Z14VelocityVerletdiP8_IO_FILE
	.type	_Z14VelocityVerletdiP8_IO_FILE, @function
_Z14VelocityVerletdiP8_IO_FILE:
.LFB5590:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	leal	(%rax,%rax,2), %r14d
	testl	%r14d, %r14d
	jle	.L115
	leal	-1(%r14), %r13d
	cmpl	$2, %r13d
	jbe	.L131
	movl	%r14d, %esi
	vmovddup	%xmm0, %xmm1
	leaq	a(%rip), %r12
	xorl	%edi, %edi
	shrl	$2, %esi
	vinsertf128	$1, %xmm1, %ymm1, %ymm1
	leaq	v(%rip), %rbx
	salq	$5, %rsi
	leaq	r(%rip), %rdx
	leaq	-32(%rsi), %rcx
	vmulpd	.LC6(%rip), %ymm1, %ymm2
	shrq	$5, %rcx
	addq	$1, %rcx
	andl	$7, %ecx
	je	.L117
	cmpq	$1, %rcx
	je	.L200
	cmpq	$2, %rcx
	je	.L201
	cmpq	$3, %rcx
	je	.L202
	cmpq	$4, %rcx
	je	.L203
	cmpq	$5, %rcx
	je	.L204
	cmpq	$6, %rcx
	jne	.L252
.L205:
	vmulpd	(%r12,%rdi), %ymm2, %ymm7
	vaddpd	(%rbx,%rdi), %ymm7, %ymm8
	vmulpd	%ymm8, %ymm1, %ymm9
	vmovapd	%ymm8, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm9, %ymm10
	vmovapd	%ymm10, (%rdx,%rdi)
	addq	$32, %rdi
.L204:
	vmulpd	(%r12,%rdi), %ymm2, %ymm11
	vaddpd	(%rbx,%rdi), %ymm11, %ymm12
	vmulpd	%ymm12, %ymm1, %ymm13
	vmovapd	%ymm12, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm13, %ymm14
	vmovapd	%ymm14, (%rdx,%rdi)
	addq	$32, %rdi
.L203:
	vmulpd	(%r12,%rdi), %ymm2, %ymm15
	vaddpd	(%rbx,%rdi), %ymm15, %ymm3
	vmulpd	%ymm3, %ymm1, %ymm5
	vmovapd	%ymm3, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm5, %ymm4
	vmovapd	%ymm4, (%rdx,%rdi)
	addq	$32, %rdi
.L202:
	vmulpd	(%r12,%rdi), %ymm2, %ymm6
	vaddpd	(%rbx,%rdi), %ymm6, %ymm7
	vmulpd	%ymm7, %ymm1, %ymm8
	vmovapd	%ymm7, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm8, %ymm9
	vmovapd	%ymm9, (%rdx,%rdi)
	addq	$32, %rdi
.L201:
	vmulpd	(%r12,%rdi), %ymm2, %ymm10
	vaddpd	(%rbx,%rdi), %ymm10, %ymm11
	vmulpd	%ymm11, %ymm1, %ymm12
	vmovapd	%ymm11, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm12, %ymm13
	vmovapd	%ymm13, (%rdx,%rdi)
	addq	$32, %rdi
.L200:
	vmulpd	(%r12,%rdi), %ymm2, %ymm14
	vaddpd	(%rbx,%rdi), %ymm14, %ymm15
	vmulpd	%ymm15, %ymm1, %ymm3
	vmovapd	%ymm15, (%rbx,%rdi)
	vaddpd	(%rdx,%rdi), %ymm3, %ymm5
	vmovapd	%ymm5, (%rdx,%rdi)
	addq	$32, %rdi
	cmpq	%rsi, %rdi
	je	.L248
.L117:
	vmulpd	(%r12,%rdi), %ymm2, %ymm4
	vmulpd	32(%rdi,%r12), %ymm2, %ymm9
	vaddpd	(%rbx,%rdi), %ymm4, %ymm6
	vaddpd	32(%rdi,%rbx), %ymm9, %ymm10
	vmulpd	64(%rdi,%r12), %ymm2, %ymm13
	vaddpd	64(%rdi,%rbx), %ymm13, %ymm14
	vmulpd	96(%rdi,%r12), %ymm2, %ymm5
	vmulpd	%ymm6, %ymm1, %ymm7
	vmovapd	%ymm6, (%rbx,%rdi)
	vaddpd	96(%rdi,%rbx), %ymm5, %ymm6
	vmulpd	%ymm10, %ymm1, %ymm11
	vaddpd	(%rdx,%rdi), %ymm7, %ymm8
	vmovapd	%ymm10, 32(%rdi,%rbx)
	vaddpd	32(%rdi,%rdx), %ymm11, %ymm12
	vmulpd	%ymm14, %ymm1, %ymm15
	vmovapd	%ymm14, 64(%rdi,%rbx)
	vaddpd	64(%rdi,%rdx), %ymm15, %ymm3
	vmulpd	%ymm6, %ymm1, %ymm4
	vmovapd	%ymm6, 96(%rdi,%rbx)
	vmovapd	%ymm8, (%rdx,%rdi)
	vaddpd	96(%rdi,%rdx), %ymm4, %ymm7
	vmulpd	128(%rdi,%r12), %ymm2, %ymm8
	addq	$256, %rdi
	vmovapd	%ymm12, -224(%rdi,%rdx)
	vmulpd	-96(%rdi,%r12), %ymm2, %ymm12
	vaddpd	-128(%rdi,%rbx), %ymm8, %ymm9
	vmovapd	%ymm3, -192(%rdi,%rdx)
	vaddpd	-96(%rdi,%rbx), %ymm12, %ymm13
	vmulpd	-64(%rdi,%r12), %ymm2, %ymm3
	vmovapd	%ymm7, -160(%rdi,%rdx)
	vaddpd	-64(%rdi,%rbx), %ymm3, %ymm5
	vmulpd	%ymm9, %ymm1, %ymm10
	vmovapd	%ymm9, -128(%rdi,%rbx)
	vmulpd	%ymm13, %ymm1, %ymm14
	vaddpd	-128(%rdi,%rdx), %ymm10, %ymm11
	vmovapd	%ymm13, -96(%rdi,%rbx)
	vaddpd	-96(%rdi,%rdx), %ymm14, %ymm15
	vmulpd	%ymm5, %ymm1, %ymm6
	vmovapd	%ymm11, -128(%rdi,%rdx)
	vmovapd	%ymm15, -96(%rdi,%rdx)
	vaddpd	-64(%rdi,%rdx), %ymm6, %ymm4
	vmulpd	-32(%rdi,%r12), %ymm2, %ymm7
	vaddpd	-32(%rdi,%rbx), %ymm7, %ymm8
	vmovapd	%ymm5, -64(%rdi,%rbx)
	vmovapd	%ymm4, -64(%rdi,%rdx)
	vmulpd	%ymm8, %ymm1, %ymm9
	vmovapd	%ymm8, -32(%rdi,%rbx)
	vaddpd	-32(%rdi,%rdx), %ymm9, %ymm10
	vmovapd	%ymm10, -32(%rdi,%rdx)
	cmpq	%rsi, %rdi
	jne	.L117
.L248:
	movl	%r14d, %eax
	andl	$-4, %eax
	cmpl	%eax, %r14d
	je	.L253
	vzeroupper
.L116:
	vmulsd	.LC1(%rip), %xmm0, %xmm11
	movslq	%eax, %r8
	leal	1(%rax), %r9d
	vmulsd	(%r12,%r8,8), %xmm11, %xmm2
	vaddsd	(%rbx,%r8,8), %xmm2, %xmm12
	vmulsd	%xmm12, %xmm0, %xmm13
	vmovsd	%xmm12, (%rbx,%r8,8)
	vaddsd	(%rdx,%r8,8), %xmm13, %xmm14
	vmovsd	%xmm14, (%rdx,%r8,8)
	cmpl	%r9d, %r14d
	jle	.L119
	movslq	%r9d, %r10
	addl	$2, %eax
	vmulsd	(%r12,%r10,8), %xmm11, %xmm15
	vaddsd	(%rbx,%r10,8), %xmm15, %xmm5
	vmulsd	%xmm5, %xmm0, %xmm3
	vmovsd	%xmm5, (%rbx,%r10,8)
	vaddsd	(%rdx,%r10,8), %xmm3, %xmm6
	vmovsd	%xmm6, (%rdx,%r10,8)
	cmpl	%eax, %r14d
	jle	.L119
	cltq
	vmulsd	(%r12,%rax,8), %xmm11, %xmm4
	vaddsd	(%rbx,%rax,8), %xmm4, %xmm7
	vmulsd	%xmm7, %xmm0, %xmm8
	vmovsd	%xmm7, (%rbx,%rax,8)
	vaddsd	(%rdx,%rax,8), %xmm8, %xmm9
	vmovsd	%xmm9, (%rdx,%rax,8)
.L119:
	vmovsd	%xmm0, 24(%rsp)
	call	_Z20computeAccelerationsv
	vmovsd	24(%rsp), %xmm0
	vmulsd	.LC1(%rip), %xmm0, %xmm1
	cmpl	$2, %r13d
	jbe	.L132
.L130:
	movl	%r14d, %r11d
	leaq	v(%rip), %rcx
	vmovddup	%xmm1, %xmm10
	shrl	$2, %r11d
	vinsertf128	$1, %xmm10, %ymm10, %ymm10
	leaq	a(%rip), %rdx
	salq	$5, %r11
	addq	%rbx, %r11
	movq	%r11, %rsi
	subq	%rcx, %rsi
	subq	$32, %rsi
	shrq	$5, %rsi
	addq	$1, %rsi
	andl	$7, %esi
	je	.L122
	cmpq	$1, %rsi
	je	.L208
	cmpq	$2, %rsi
	je	.L209
	cmpq	$3, %rsi
	je	.L210
	cmpq	$4, %rsi
	je	.L211
	cmpq	$5, %rsi
	je	.L212
	cmpq	$6, %rsi
	jne	.L254
.L213:
	vmulpd	(%rdx), %ymm10, %ymm12
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm12, %ymm13
	vmovapd	%ymm13, -32(%rcx)
.L212:
	vmulpd	(%rdx), %ymm10, %ymm14
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm14, %ymm15
	vmovapd	%ymm15, -32(%rcx)
.L211:
	vmulpd	(%rdx), %ymm10, %ymm5
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm5, %ymm3
	vmovapd	%ymm3, -32(%rcx)
.L210:
	vmulpd	(%rdx), %ymm10, %ymm6
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm6, %ymm4
	vmovapd	%ymm4, -32(%rcx)
.L209:
	vmulpd	(%rdx), %ymm10, %ymm7
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm7, %ymm8
	vmovapd	%ymm8, -32(%rcx)
.L208:
	vmulpd	(%rdx), %ymm10, %ymm9
	addq	$32, %rcx
	addq	$32, %rdx
	vaddpd	-32(%rcx), %ymm9, %ymm11
	vmovapd	%ymm11, -32(%rcx)
	cmpq	%r11, %rcx
	je	.L158
.L122:
	vmulpd	(%rdx), %ymm10, %ymm2
	addq	$256, %rcx
	addq	$256, %rdx
	vaddpd	-256(%rcx), %ymm2, %ymm12
	vmovapd	%ymm12, -256(%rcx)
	vmulpd	-224(%rdx), %ymm10, %ymm13
	vaddpd	-224(%rcx), %ymm13, %ymm14
	vmovapd	%ymm14, -224(%rcx)
	vmulpd	-192(%rdx), %ymm10, %ymm15
	vaddpd	-192(%rcx), %ymm15, %ymm5
	vmovapd	%ymm5, -192(%rcx)
	vmulpd	-160(%rdx), %ymm10, %ymm3
	vaddpd	-160(%rcx), %ymm3, %ymm6
	vmovapd	%ymm6, -160(%rcx)
	vmulpd	-128(%rdx), %ymm10, %ymm4
	vaddpd	-128(%rcx), %ymm4, %ymm7
	vmovapd	%ymm7, -128(%rcx)
	vmulpd	-96(%rdx), %ymm10, %ymm8
	vaddpd	-96(%rcx), %ymm8, %ymm9
	vmovapd	%ymm9, -96(%rcx)
	vmulpd	-64(%rdx), %ymm10, %ymm11
	vaddpd	-64(%rcx), %ymm11, %ymm2
	vmovapd	%ymm2, -64(%rcx)
	vmulpd	-32(%rdx), %ymm10, %ymm12
	vaddpd	-32(%rcx), %ymm12, %ymm13
	vmovapd	%ymm13, -32(%rcx)
	cmpq	%r11, %rcx
	jne	.L122
.L158:
	movl	%r14d, %eax
	andl	$-4, %eax
	testb	$3, %r14b
	je	.L255
	vzeroupper
.L129:
	movslq	%eax, %rdi
	leal	1(%rax), %r8d
	vmulsd	(%r12,%rdi,8), %xmm1, %xmm10
	vaddsd	(%rbx,%rdi,8), %xmm10, %xmm14
	vmovsd	%xmm14, (%rbx,%rdi,8)
	cmpl	%r14d, %r8d
	jge	.L120
	movslq	%r8d, %r9
	addl	$2, %eax
	vmulsd	(%r12,%r9,8), %xmm1, %xmm15
	vaddsd	(%rbx,%r9,8), %xmm15, %xmm5
	vmovsd	%xmm5, (%rbx,%r9,8)
	cmpl	%r14d, %eax
	jge	.L120
	cltq
	vmulsd	(%r12,%rax,8), %xmm1, %xmm1
	vaddsd	(%rbx,%rax,8), %xmm1, %xmm3
	vmovsd	%xmm3, (%rbx,%rax,8)
.L120:
	leaq	v(%rip), %rcx
	movl	%r13d, %r14d
	vmovsd	m(%rip), %xmm6
	vmovq	.LC7(%rip), %xmm12
	leaq	8(%rcx), %r13
	vmovq	.LC8(%rip), %xmm13
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovsd	L(%rip), %xmm11
	leaq	0(%r13,%r14,8), %r12
	vaddsd	%xmm6, %xmm6, %xmm4
	vmovapd	%xmm2, %xmm1
	movq	%r12, %rbx
	leaq	r(%rip), %rsi
	vmovapd	%xmm12, %xmm7
	subq	%rcx, %rbx
	vmovapd	%xmm13, %xmm8
	vdivsd	%xmm0, %xmm4, %xmm0
	subq	$8, %rbx
	shrq	$3, %rbx
	addq	$1, %rbx
	andl	$3, %ebx
	je	.L127
	cmpq	$1, %rbx
	je	.L206
	cmpq	$2, %rbx
	je	.L207
	vmovsd	(%rsi), %xmm10
	vcomisd	%xmm10, %xmm2
	ja	.L256
.L236:
	vcomisd	%xmm11, %xmm10
	jnb	.L257
.L237:
	addq	$8, %rsi
	addq	$8, %rcx
.L207:
	vmovsd	(%rsi), %xmm15
	vcomisd	%xmm15, %xmm2
	jbe	.L238
	vmovsd	(%rcx), %xmm9
	vandpd	%xmm13, %xmm9, %xmm6
	vmovapd	%xmm9, %xmm5
	vmulsd	%xmm0, %xmm6, %xmm4
	vxorpd	%xmm12, %xmm5, %xmm3
	vmovsd	%xmm3, (%rcx)
	vaddsd	%xmm4, %xmm1, %xmm1
.L238:
	vcomisd	%xmm11, %xmm15
	jb	.L239
	vmovsd	(%rcx), %xmm10
	vandpd	%xmm8, %xmm10, %xmm9
	vmovapd	%xmm10, %xmm14
	vmulsd	%xmm9, %xmm0, %xmm5
	vxorpd	%xmm7, %xmm14, %xmm15
	vmovsd	%xmm15, (%rcx)
	vaddsd	%xmm5, %xmm1, %xmm1
.L239:
	addq	$8, %rsi
	addq	$8, %rcx
.L206:
	vmovsd	(%rsi), %xmm3
	vcomisd	%xmm3, %xmm2
	ja	.L258
	vcomisd	%xmm11, %xmm3
	jb	.L241
.L259:
	vmovsd	(%rcx), %xmm5
	vandpd	%xmm8, %xmm5, %xmm6
	vmovapd	%xmm5, %xmm9
	vmulsd	%xmm6, %xmm0, %xmm4
	vxorpd	%xmm7, %xmm9, %xmm3
	vmovsd	%xmm3, (%rcx)
	vaddsd	%xmm4, %xmm1, %xmm1
.L241:
	addq	$8, %rcx
	addq	$8, %rsi
	cmpq	%r12, %rcx
	je	.L242
.L127:
	vmovsd	(%rsi), %xmm14
	vcomisd	%xmm14, %xmm2
	jbe	.L123
	vmovsd	(%rcx), %xmm15
	vandpd	%xmm13, %xmm15, %xmm9
	vmovapd	%xmm15, %xmm10
	vmulsd	%xmm0, %xmm9, %xmm3
	vxorpd	%xmm12, %xmm10, %xmm5
	vmovsd	%xmm5, (%rcx)
	vaddsd	%xmm3, %xmm1, %xmm1
.L123:
	vcomisd	%xmm11, %xmm14
	jb	.L125
	vmovsd	(%rcx), %xmm6
	vandpd	%xmm8, %xmm6, %xmm15
	vmovapd	%xmm6, %xmm4
	vmulsd	%xmm15, %xmm0, %xmm10
	vxorpd	%xmm7, %xmm4, %xmm14
	vmovsd	%xmm14, (%rcx)
	vaddsd	%xmm10, %xmm1, %xmm1
.L125:
	vmovsd	8(%rsi), %xmm5
	leaq	8(%rsi), %r10
	leaq	8(%rcx), %r11
	vcomisd	%xmm5, %xmm2
	jbe	.L233
	vmovsd	8(%rcx), %xmm9
	vandpd	%xmm13, %xmm9, %xmm4
	vmovapd	%xmm9, %xmm3
	vmulsd	%xmm0, %xmm4, %xmm14
	vxorpd	%xmm12, %xmm3, %xmm6
	vmovsd	%xmm6, 8(%rcx)
	vaddsd	%xmm14, %xmm1, %xmm1
.L233:
	vcomisd	%xmm11, %xmm5
	jb	.L243
	vmovsd	(%r11), %xmm15
	vandpd	%xmm8, %xmm15, %xmm9
	vmovapd	%xmm15, %xmm10
	vmulsd	%xmm9, %xmm0, %xmm3
	vxorpd	%xmm7, %xmm10, %xmm5
	vmovsd	%xmm5, (%r11)
	vaddsd	%xmm3, %xmm1, %xmm1
.L243:
	vmovsd	8(%r10), %xmm6
	vcomisd	%xmm6, %xmm2
	jbe	.L244
	vmovsd	8(%r11), %xmm4
	vandpd	%xmm13, %xmm4, %xmm10
	vmovapd	%xmm4, %xmm14
	vmulsd	%xmm0, %xmm10, %xmm5
	vxorpd	%xmm12, %xmm14, %xmm15
	vmovsd	%xmm15, 8(%r11)
	vaddsd	%xmm5, %xmm1, %xmm1
.L244:
	vcomisd	%xmm11, %xmm6
	jb	.L245
	vmovsd	8(%r11), %xmm3
	vandpd	%xmm8, %xmm3, %xmm4
	vmovapd	%xmm3, %xmm9
	vmulsd	%xmm4, %xmm0, %xmm14
	vxorpd	%xmm7, %xmm9, %xmm6
	vmovsd	%xmm6, 8(%r11)
	vaddsd	%xmm14, %xmm1, %xmm1
.L245:
	vmovsd	16(%r10), %xmm15
	vcomisd	%xmm15, %xmm2
	jbe	.L246
	vmovsd	16(%r11), %xmm5
	vandpd	%xmm13, %xmm5, %xmm9
	vmovapd	%xmm5, %xmm10
	vmulsd	%xmm0, %xmm9, %xmm6
	vxorpd	%xmm12, %xmm10, %xmm3
	vmovsd	%xmm3, 16(%r11)
	vaddsd	%xmm6, %xmm1, %xmm1
.L246:
	vcomisd	%xmm11, %xmm15
	jb	.L247
	vmovsd	16(%r11), %xmm4
	vandpd	%xmm8, %xmm4, %xmm5
	vmovapd	%xmm4, %xmm14
	vmulsd	%xmm5, %xmm0, %xmm10
	vxorpd	%xmm7, %xmm14, %xmm15
	vmovsd	%xmm15, 16(%r11)
	vaddsd	%xmm10, %xmm1, %xmm1
.L247:
	leaq	24(%r11), %rcx
	leaq	24(%r10), %rsi
	cmpq	%r12, %rcx
	jne	.L127
.L242:
	vmulsd	.LC9(%rip), %xmm1, %xmm12
.L128:
	vmulsd	%xmm11, %xmm11, %xmm11
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	vdivsd	%xmm11, %xmm12, %xmm0
	ret
	.p2align 4,,10
	.p2align 3
.L258:
	.cfi_restore_state
	vmovsd	(%rcx), %xmm6
	vcomisd	%xmm11, %xmm3
	vandpd	%xmm13, %xmm6, %xmm14
	vmovapd	%xmm6, %xmm4
	vmulsd	%xmm0, %xmm14, %xmm15
	vxorpd	%xmm12, %xmm4, %xmm10
	vmovsd	%xmm10, (%rcx)
	vaddsd	%xmm15, %xmm1, %xmm1
	jb	.L241
	jmp	.L259
	.p2align 4,,10
	.p2align 3
.L253:
	vmovsd	%xmm0, 24(%rsp)
	vzeroupper
	call	_Z20computeAccelerationsv
	vmovsd	24(%rsp), %xmm0
	vmulsd	.LC1(%rip), %xmm0, %xmm1
	jmp	.L130
	.p2align 4,,10
	.p2align 3
.L255:
	vzeroupper
	jmp	.L120
	.p2align 4,,10
	.p2align 3
.L257:
	vmovsd	(%rcx), %xmm3
	vandpd	%xmm8, %xmm3, %xmm10
	vmovapd	%xmm3, %xmm6
	vmulsd	%xmm10, %xmm0, %xmm14
	vxorpd	%xmm7, %xmm6, %xmm4
	vmovsd	%xmm4, (%rcx)
	vaddsd	%xmm14, %xmm1, %xmm1
	jmp	.L237
	.p2align 4,,10
	.p2align 3
.L254:
	vmulpd	(%rdx), %ymm10, %ymm11
	leaq	32+a(%rip), %rdx
	vaddpd	(%rcx), %ymm11, %ymm2
	vmovapd	%ymm2, (%rcx)
	leaq	32+v(%rip), %rcx
	jmp	.L213
	.p2align 4,,10
	.p2align 3
.L252:
	vmulpd	(%r12), %ymm2, %ymm3
	movl	$32, %edi
	vaddpd	(%rbx), %ymm3, %ymm5
	vmulpd	%ymm5, %ymm1, %ymm4
	vmovapd	%ymm5, (%rbx)
	vaddpd	(%rdx), %ymm4, %ymm6
	vmovapd	%ymm6, (%rdx)
	jmp	.L205
	.p2align 4,,10
	.p2align 3
.L256:
	vmovsd	(%rcx), %xmm14
	vmovapd	%xmm14, %xmm9
	vandpd	%xmm13, %xmm14, %xmm5
	vmulsd	%xmm0, %xmm5, %xmm1
	vxorpd	%xmm12, %xmm9, %xmm15
	vmovsd	%xmm15, (%rcx)
	jmp	.L236
	.p2align 4,,10
	.p2align 3
.L115:
	call	_Z20computeAccelerationsv
	vmovsd	L(%rip), %xmm11
	vxorpd	%xmm12, %xmm12, %xmm12
	jmp	.L128
.L131:
	xorl	%eax, %eax
	leaq	a(%rip), %r12
	leaq	v(%rip), %rbx
	leaq	r(%rip), %rdx
	jmp	.L116
.L132:
	xorl	%eax, %eax
	jmp	.L129
	.cfi_endproc
.LFE5590:
	.size	_Z14VelocityVerletdiP8_IO_FILE, .-_Z14VelocityVerletdiP8_IO_FILE
	.p2align 4
	.globl	_Z20initializeVelocitiesv
	.type	_Z20initializeVelocitiesv, @function
_Z20initializeVelocitiesv:
.LFB5591:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$32, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	leal	(%rax,%rax,2), %r15d
	testl	%r15d, %r15d
	jle	.L401
	leaq	v(%rip), %rbx
	movzbl	_ZZ9gaussdistvE9available(%rip), %esi
	xorl	%r14d, %r14d
	movq	%rbx, %r13
	movq	%rbx, %r12
	.p2align 4,,10
	.p2align 3
.L267:
	testb	%sil, %sil
	je	.L354
	vmovsd	_ZZ9gaussdistvE4gset(%rip), %xmm0
	movb	$0, _ZZ9gaussdistvE9available(%rip)
	vmovsd	%xmm0, (%r12)
	.p2align 4,,10
	.p2align 3
.L355:
	call	rand@PLT
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtsi2sdl	%eax, %xmm5, %xmm1
	vmulsd	.LC10(%rip), %xmm1, %xmm2
	vsubsd	.LC2(%rip), %xmm2, %xmm3
	vmovsd	%xmm3, -56(%rbp)
	call	rand@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovsd	-56(%rbp), %xmm9
	vmovsd	.LC2(%rip), %xmm13
	vcvtsi2sdl	%eax, %xmm4, %xmm6
	vmulsd	.LC10(%rip), %xmm6, %xmm7
	vsubsd	.LC2(%rip), %xmm7, %xmm8
	vmulsd	%xmm9, %xmm9, %xmm10
	vmulsd	%xmm8, %xmm8, %xmm11
	vaddsd	%xmm11, %xmm10, %xmm12
	vcomisd	%xmm12, %xmm13
	jbe	.L355
	vxorpd	%xmm14, %xmm14, %xmm14
	vcomisd	%xmm14, %xmm12
	je	.L355
	vmovapd	%xmm12, %xmm0
	vmovsd	%xmm8, -72(%rbp)
	vmovsd	%xmm9, -64(%rbp)
	vmovsd	%xmm12, -56(%rbp)
	call	log@PLT
	vmulsd	.LC11(%rip), %xmm0, %xmm15
	xorl	%esi, %esi
	vmovsd	-56(%rbp), %xmm0
	vmovsd	-64(%rbp), %xmm1
	vmovsd	-72(%rbp), %xmm2
	movb	$0, _ZZ9gaussdistvE9available(%rip)
	vdivsd	%xmm0, %xmm15, %xmm5
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vmulsd	%xmm5, %xmm1, %xmm14
	vmulsd	%xmm5, %xmm2, %xmm3
	vmovsd	%xmm14, _ZZ9gaussdistvE4gset(%rip)
	vmovsd	%xmm3, 8(%r12)
	jmp	.L266
	.p2align 4,,10
	.p2align 3
.L354:
	call	rand@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sdl	%eax, %xmm4, %xmm6
	vmulsd	.LC10(%rip), %xmm6, %xmm7
	vsubsd	.LC2(%rip), %xmm7, %xmm8
	vmovsd	%xmm8, -56(%rbp)
	call	rand@PLT
	vxorpd	%xmm9, %xmm9, %xmm9
	vmovsd	-56(%rbp), %xmm13
	vmovsd	.LC2(%rip), %xmm0
	vcvtsi2sdl	%eax, %xmm9, %xmm10
	vmulsd	.LC10(%rip), %xmm10, %xmm11
	vsubsd	.LC2(%rip), %xmm11, %xmm12
	vmulsd	%xmm13, %xmm13, %xmm14
	vmulsd	%xmm12, %xmm12, %xmm15
	vaddsd	%xmm15, %xmm14, %xmm5
	vcomisd	%xmm5, %xmm0
	jbe	.L354
	vxorpd	%xmm1, %xmm1, %xmm1
	vcomisd	%xmm1, %xmm5
	je	.L354
	vmovapd	%xmm5, %xmm0
	vmovsd	%xmm12, -72(%rbp)
	vmovsd	%xmm13, -64(%rbp)
	vmovsd	%xmm5, -56(%rbp)
	call	log@PLT
	vmovsd	-56(%rbp), %xmm3
	vmovsd	-64(%rbp), %xmm6
	movb	$0, _ZZ9gaussdistvE9available(%rip)
	vmulsd	.LC11(%rip), %xmm0, %xmm2
	vmovsd	-72(%rbp), %xmm8
	vdivsd	%xmm3, %xmm2, %xmm4
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vmulsd	%xmm4, %xmm6, %xmm7
	vmulsd	%xmm4, %xmm8, %xmm9
	vmovsd	%xmm7, _ZZ9gaussdistvE4gset(%rip)
	vmovsd	%xmm9, (%r12)
	vmovsd	%xmm7, 8(%r12)
	.p2align 4,,10
	.p2align 3
.L264:
	call	rand@PLT
	vxorpd	%xmm10, %xmm10, %xmm10
	vcvtsi2sdl	%eax, %xmm10, %xmm11
	vmulsd	.LC10(%rip), %xmm11, %xmm12
	vsubsd	.LC2(%rip), %xmm12, %xmm13
	vmovsd	%xmm13, -56(%rbp)
	call	rand@PLT
	vxorpd	%xmm14, %xmm14, %xmm14
	vmovsd	-56(%rbp), %xmm2
	vmovsd	.LC2(%rip), %xmm4
	vcvtsi2sdl	%eax, %xmm14, %xmm15
	vmulsd	.LC10(%rip), %xmm15, %xmm5
	vxorpd	%xmm6, %xmm6, %xmm6
	vsubsd	.LC2(%rip), %xmm5, %xmm1
	vmulsd	%xmm2, %xmm2, %xmm3
	vmulsd	%xmm1, %xmm1, %xmm0
	vaddsd	%xmm0, %xmm3, %xmm7
	vcomisd	%xmm7, %xmm4
	seta	%cl
	vcomisd	%xmm6, %xmm7
	setne	%dl
	andb	%dl, %cl
	je	.L264
	vmovapd	%xmm7, %xmm0
	movb	%cl, -73(%rbp)
	vmovsd	%xmm1, -72(%rbp)
	vmovsd	%xmm2, -64(%rbp)
	vmovsd	%xmm7, -56(%rbp)
	call	log@PLT
	vmovsd	-56(%rbp), %xmm9
	vmovsd	-64(%rbp), %xmm11
	movb	$1, _ZZ9gaussdistvE9available(%rip)
	vmulsd	.LC11(%rip), %xmm0, %xmm8
	vmovsd	-72(%rbp), %xmm13
	movzbl	-73(%rbp), %esi
	vdivsd	%xmm9, %xmm8, %xmm10
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vmulsd	%xmm10, %xmm11, %xmm12
	vmulsd	%xmm10, %xmm13, %xmm14
	vmovsd	%xmm12, _ZZ9gaussdistvE4gset(%rip)
.L266:
	addl	$3, %r14d
	vmovsd	%xmm14, 16(%r12)
	addq	$24, %r12
	cmpl	%r14d, %r15d
	jg	.L267
	leal	-1(%r15), %r11d
	movl	$2863311531, %edi
	movq	%r11, %r10
	imulq	%rdi, %r11
	shrq	$33, %r11
	addl	$1, %r11d
	cmpl	$8, %r10d
	jbe	.L288
	movl	%r11d, %r8d
	leaq	v(%rip), %rcx
	vxorpd	%xmm5, %xmm5, %xmm5
	movabsq	$192153584101141163, %r14
	shrl	$2, %r8d
	vmovapd	%ymm5, %ymm0
	vmovapd	%ymm5, %ymm7
	leaq	(%r8,%r8,2), %r9
	salq	$5, %r9
	addq	%rbx, %r9
	movq	%r9, %r12
	subq	%rcx, %r12
	subq	$96, %r12
	shrq	$5, %r12
	imulq	%r14, %r12
	addq	$1, %r12
	andl	$7, %r12d
	je	.L269
	cmpq	$1, %r12
	je	.L356
	cmpq	$2, %r12
	je	.L357
	cmpq	$3, %r12
	je	.L358
	cmpq	$4, %r12
	je	.L359
	cmpq	$5, %r12
	je	.L360
	cmpq	$6, %r12
	jne	.L402
.L361:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
.L360:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
.L359:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
.L358:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
.L357:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
.L356:
	vaddpd	(%rcx), %ymm7, %ymm7
	vaddpd	32(%rcx), %ymm0, %ymm0
	addq	$96, %rcx
	vaddpd	-32(%rcx), %ymm5, %ymm5
	cmpq	%r9, %rcx
	je	.L398
.L269:
	vaddpd	(%rcx), %ymm7, %ymm3
	vaddpd	32(%rcx), %ymm0, %ymm4
	addq	$768, %rcx
	vaddpd	-704(%rcx), %ymm5, %ymm6
	vaddpd	-672(%rcx), %ymm3, %ymm8
	vaddpd	-640(%rcx), %ymm4, %ymm9
	vaddpd	-608(%rcx), %ymm6, %ymm10
	vaddpd	-576(%rcx), %ymm8, %ymm11
	vaddpd	-544(%rcx), %ymm9, %ymm12
	vaddpd	-512(%rcx), %ymm10, %ymm13
	vaddpd	-480(%rcx), %ymm11, %ymm14
	vaddpd	-448(%rcx), %ymm12, %ymm15
	vaddpd	-416(%rcx), %ymm13, %ymm2
	vaddpd	-352(%rcx), %ymm15, %ymm7
	vaddpd	-320(%rcx), %ymm2, %ymm5
	vaddpd	-256(%rcx), %ymm7, %ymm0
	vaddpd	-224(%rcx), %ymm5, %ymm4
	vaddpd	-384(%rcx), %ymm14, %ymm1
	vaddpd	-160(%rcx), %ymm0, %ymm8
	vaddpd	-288(%rcx), %ymm1, %ymm3
	vaddpd	-128(%rcx), %ymm4, %ymm9
	vaddpd	-192(%rcx), %ymm3, %ymm6
	vaddpd	-64(%rcx), %ymm8, %ymm0
	vaddpd	-96(%rcx), %ymm6, %ymm7
	vaddpd	-32(%rcx), %ymm9, %ymm5
	cmpq	%r9, %rcx
	jne	.L269
.L398:
	vextractf128	$0x1, %ymm7, %xmm12
	vunpckhpd	%xmm7, %xmm7, %xmm15
	vmovapd	%xmm5, %xmm3
	movl	%r11d, %edx
	vunpckhpd	%xmm12, %xmm12, %xmm13
	vaddsd	%xmm0, %xmm15, %xmm6
	vextractf128	$0x1, %ymm0, %xmm4
	andl	$-4, %edx
	vaddsd	%xmm13, %xmm7, %xmm14
	vunpckhpd	%xmm0, %xmm0, %xmm7
	vunpckhpd	%xmm5, %xmm5, %xmm0
	vextractf128	$0x1, %ymm5, %xmm5
	vunpckhpd	%xmm4, %xmm4, %xmm11
	vaddsd	%xmm7, %xmm12, %xmm8
	vunpckhpd	%xmm5, %xmm5, %xmm15
	vaddsd	%xmm0, %xmm4, %xmm10
	vaddsd	%xmm5, %xmm11, %xmm13
	vaddsd	%xmm15, %xmm3, %xmm1
	leal	(%rdx,%rdx,2), %eax
	vaddsd	%xmm10, %xmm14, %xmm2
	vaddsd	%xmm13, %xmm6, %xmm15
	vaddsd	%xmm1, %xmm8, %xmm1
	cmpl	%edx, %r11d
	je	.L403
.L268:
	leal	1(%rax), %r8d
	leal	2(%rax), %r12d
	movslq	%eax, %rdi
	movslq	%r8d, %r9
	leal	3(%rax), %ecx
	vaddsd	0(%r13,%rdi,8), %xmm2, %xmm4
	movslq	%r12d, %r14
	vaddsd	0(%r13,%r9,8), %xmm15, %xmm9
	vaddsd	0(%r13,%r14,8), %xmm1, %xmm10
	cmpl	%ecx, %r15d
	jle	.L271
	leal	4(%rax), %esi
	leal	5(%rax), %r8d
	movslq	%ecx, %rdx
	movslq	%esi, %rdi
	movslq	%r8d, %r9
	leal	6(%rax), %r12d
	vaddsd	0(%r13,%rdx,8), %xmm4, %xmm4
	vaddsd	0(%r13,%rdi,8), %xmm9, %xmm9
	vaddsd	0(%r13,%r9,8), %xmm10, %xmm10
	cmpl	%r12d, %r15d
	jle	.L271
	leal	7(%rax), %ecx
	addl	$8, %eax
	movslq	%r12d, %r14
	movslq	%ecx, %rdx
	cltq
	vaddsd	0(%r13,%r14,8), %xmm4, %xmm4
	vaddsd	0(%r13,%rdx,8), %xmm9, %xmm9
	vaddsd	0(%r13,%rax,8), %xmm10, %xmm10
.L271:
	movl	N(%rip), %esi
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	.LC2(%rip), %xmm5
	vcvtsi2sdl	%esi, %xmm3, %xmm11
	vdivsd	%xmm11, %xmm5, %xmm12
	vmulsd	%xmm4, %xmm12, %xmm2
	vmulsd	%xmm9, %xmm12, %xmm0
	vmulsd	%xmm10, %xmm12, %xmm1
	cmpl	$8, %r10d
	jbe	.L291
.L287:
	vunpcklpd	%xmm1, %xmm0, %xmm13
	vunpcklpd	%xmm2, %xmm1, %xmm15
	vunpcklpd	%xmm0, %xmm2, %xmm14
	movl	%r11d, %eax
	shrl	$2, %eax
	leaq	v(%rip), %r12
	movabsq	$192153584101141163, %r9
	leaq	(%rax,%rax,2), %rdi
	vinsertf128	$0x1, %xmm13, %ymm15, %ymm5
	vinsertf128	$0x1, %xmm14, %ymm13, %ymm3
	salq	$5, %rdi
	vinsertf128	$0x1, %xmm15, %ymm14, %ymm4
	addq	%rbx, %rdi
	movq	%rdi, %r8
	subq	%r12, %r8
	subq	$96, %r8
	shrq	$5, %r8
	imulq	%r9, %r8
	addq	$1, %r8
	andl	$3, %r8d
	je	.L275
	cmpq	$1, %r8
	je	.L368
	cmpq	$2, %r8
	je	.L369
	vmovapd	(%r12), %ymm10
	vmovapd	32+v(%rip), %ymm6
	vmovapd	64+v(%rip), %ymm8
	vsubpd	%ymm3, %ymm6, %ymm7
	vsubpd	%ymm4, %ymm10, %ymm11
	vsubpd	%ymm5, %ymm8, %ymm9
	vmovapd	%ymm11, (%r12)
	leaq	96+v(%rip), %r12
	vmovapd	%ymm7, 32+v(%rip)
	vmovapd	%ymm9, 64+v(%rip)
.L369:
	vmovapd	32(%r12), %ymm12
	vmovapd	64(%r12), %ymm15
	addq	$96, %r12
	vmovapd	-96(%r12), %ymm14
	vsubpd	%ymm3, %ymm12, %ymm13
	vsubpd	%ymm5, %ymm15, %ymm6
	vsubpd	%ymm4, %ymm14, %ymm7
	vmovapd	%ymm13, -64(%r12)
	vmovapd	%ymm7, -96(%r12)
	vmovapd	%ymm6, -32(%r12)
.L368:
	vmovapd	32(%r12), %ymm8
	vmovapd	64(%r12), %ymm10
	addq	$96, %r12
	vmovapd	-96(%r12), %ymm12
	vsubpd	%ymm3, %ymm8, %ymm9
	vsubpd	%ymm5, %ymm10, %ymm11
	vsubpd	%ymm4, %ymm12, %ymm13
	vmovapd	%ymm9, -64(%r12)
	vmovapd	%ymm13, -96(%r12)
	vmovapd	%ymm11, -32(%r12)
	cmpq	%rdi, %r12
	je	.L324
.L275:
	vmovapd	32(%r12), %ymm15
	vmovapd	(%r12), %ymm8
	addq	$384, %r12
	vmovapd	-320(%r12), %ymm6
	vmovapd	-256(%r12), %ymm10
	vsubpd	%ymm3, %ymm15, %ymm14
	vsubpd	%ymm4, %ymm8, %ymm9
	vmovapd	-128(%r12), %ymm8
	vmovapd	-224(%r12), %ymm12
	vsubpd	%ymm5, %ymm6, %ymm7
	vsubpd	%ymm3, %ymm10, %ymm11
	vmovapd	-288(%r12), %ymm15
	vmovapd	-160(%r12), %ymm6
	vmovapd	-192(%r12), %ymm10
	vsubpd	%ymm5, %ymm12, %ymm13
	vmovapd	-64(%r12), %ymm12
	vmovapd	%ymm14, -352(%r12)
	vsubpd	%ymm4, %ymm15, %ymm14
	vmovapd	-32(%r12), %ymm15
	vmovapd	%ymm7, -320(%r12)
	vsubpd	%ymm3, %ymm6, %ymm7
	vmovapd	-96(%r12), %ymm6
	vmovapd	%ymm9, -384(%r12)
	vsubpd	%ymm5, %ymm8, %ymm9
	vmovapd	%ymm14, -288(%r12)
	vsubpd	%ymm5, %ymm15, %ymm14
	vmovapd	%ymm11, -256(%r12)
	vsubpd	%ymm4, %ymm10, %ymm11
	vmovapd	%ymm13, -224(%r12)
	vsubpd	%ymm3, %ymm12, %ymm13
	vmovapd	%ymm7, -160(%r12)
	vsubpd	%ymm4, %ymm6, %ymm7
	vmovapd	%ymm11, -192(%r12)
	vmovapd	%ymm9, -128(%r12)
	vmovapd	%ymm7, -96(%r12)
	vmovapd	%ymm13, -64(%r12)
	vmovapd	%ymm14, -32(%r12)
	cmpq	%rdi, %r12
	jne	.L275
.L324:
	movl	%r11d, %r14d
	andl	$-4, %r14d
	leal	(%r14,%r14,2), %eax
	cmpl	%r11d, %r14d
	je	.L272
.L285:
	movslq	%eax, %rcx
	leal	1(%rax), %edx
	leal	2(%rax), %r8d
	vmovsd	0(%r13,%rcx,8), %xmm5
	movslq	%edx, %rdi
	movslq	%r8d, %r9
	leal	3(%rax), %r12d
	vsubsd	%xmm2, %xmm5, %xmm3
	vmovsd	%xmm3, 0(%r13,%rcx,8)
	vmovsd	0(%r13,%rdi,8), %xmm4
	vsubsd	%xmm0, %xmm4, %xmm8
	vmovsd	%xmm8, 0(%r13,%rdi,8)
	vmovsd	0(%r13,%r9,8), %xmm9
	vsubsd	%xmm1, %xmm9, %xmm10
	vmovsd	%xmm10, 0(%r13,%r9,8)
	cmpl	%r12d, %r15d
	jle	.L273
	movslq	%r12d, %r14
	leal	4(%rax), %ecx
	leal	5(%rax), %edi
	vmovsd	0(%r13,%r14,8), %xmm11
	movslq	%ecx, %rdx
	movslq	%edi, %r8
	leal	6(%rax), %r9d
	vsubsd	%xmm2, %xmm11, %xmm12
	vmovsd	%xmm12, 0(%r13,%r14,8)
	vmovsd	0(%r13,%rdx,8), %xmm13
	vsubsd	%xmm0, %xmm13, %xmm15
	vmovsd	%xmm15, 0(%r13,%rdx,8)
	vmovsd	0(%r13,%r8,8), %xmm14
	vsubsd	%xmm1, %xmm14, %xmm6
	vmovsd	%xmm6, 0(%r13,%r8,8)
	cmpl	%r9d, %r15d
	jle	.L273
	movslq	%r9d, %r12
	leal	7(%rax), %r14d
	addl	$8, %eax
	vmovsd	0(%r13,%r12,8), %xmm7
	movslq	%r14d, %rcx
	cltq
	vsubsd	%xmm2, %xmm7, %xmm2
	vmovsd	%xmm2, 0(%r13,%r12,8)
	vmovsd	0(%r13,%rcx,8), %xmm5
	vsubsd	%xmm0, %xmm5, %xmm0
	vmovsd	%xmm0, 0(%r13,%rcx,8)
	vmovsd	0(%r13,%rax,8), %xmm3
	vsubsd	%xmm1, %xmm3, %xmm1
	vmovsd	%xmm1, 0(%r13,%rax,8)
.L273:
	cmpl	$8, %r10d
	jbe	.L289
.L272:
	movl	%r11d, %edx
	leaq	v(%rip), %r9
	vxorpd	%xmm10, %xmm10, %xmm10
	shrl	$2, %edx
	leaq	(%rdx,%rdx,2), %rdi
	salq	$5, %rdi
	addq	%rbx, %rdi
	movq	%rdi, %r8
	subq	%r9, %r8
	andl	$32, %r8d
	je	.L277
	vmovapd	(%r9), %ymm8
	vmovapd	64+v(%rip), %ymm9
	leaq	96+v(%rip), %r9
	vperm2f128	$48, 32+v(%rip), %ymm8, %ymm10
	vperm2f128	$2, 32+v(%rip), %ymm8, %ymm14
	vperm2f128	$33, %ymm8, %ymm8, %ymm11
	vpermilpd	$2, %ymm9, %ymm7
	vshufpd	$2, %ymm11, %ymm10, %ymm12
	vperm2f128	$33, %ymm9, %ymm12, %ymm13
	vshufpd	$5, %ymm14, %ymm10, %ymm6
	vblendpd	$8, %ymm7, %ymm6, %ymm2
	vmulpd	%ymm2, %ymm2, %ymm4
	vblendpd	$8, %ymm13, %ymm12, %ymm15
	vshufpd	$2, %ymm14, %ymm11, %ymm5
	vinsertf128	$1, %xmm9, %ymm5, %ymm0
	vmulpd	%ymm15, %ymm15, %ymm1
	vblendpd	$7, %ymm0, %ymm9, %ymm3
	vmulpd	%ymm3, %ymm3, %ymm9
	vaddpd	%ymm4, %ymm1, %ymm8
	vaddpd	%ymm9, %ymm8, %ymm10
	cmpq	%rdi, %r9
	je	.L397
	.p2align 4,,10
	.p2align 3
.L277:
	vmovapd	(%r9), %ymm11
	vmovapd	64(%r9), %ymm12
	addq	$192, %r9
	vperm2f128	$48, -160(%r9), %ymm11, %ymm13
	vperm2f128	$2, -160(%r9), %ymm11, %ymm2
	vperm2f128	$33, %ymm11, %ymm11, %ymm15
	vpermilpd	$2, %ymm12, %ymm0
	vshufpd	$2, %ymm15, %ymm13, %ymm14
	vshufpd	$2, %ymm2, %ymm15, %ymm1
	vperm2f128	$33, %ymm12, %ymm14, %ymm6
	vinsertf128	$1, %xmm12, %ymm1, %ymm4
	vblendpd	$8, %ymm6, %ymm14, %ymm7
	vblendpd	$7, %ymm4, %ymm12, %ymm8
	vshufpd	$5, %ymm2, %ymm13, %ymm5
	vblendpd	$8, %ymm0, %ymm5, %ymm3
	vmulpd	%ymm7, %ymm7, %ymm9
	vmovapd	-96(%r9), %ymm14
	vmovapd	-64(%r9), %ymm6
	vmulpd	%ymm3, %ymm3, %ymm11
	vmovapd	-32(%r9), %ymm0
	vmulpd	%ymm8, %ymm8, %ymm13
	vperm2f128	$48, %ymm6, %ymm14, %ymm1
	vperm2f128	$2, %ymm6, %ymm14, %ymm8
	vperm2f128	$33, %ymm14, %ymm14, %ymm5
	vshufpd	$5, %ymm8, %ymm1, %ymm4
	vshufpd	$2, %ymm5, %ymm1, %ymm2
	vperm2f128	$33, %ymm0, %ymm2, %ymm7
	vblendpd	$8, %ymm7, %ymm2, %ymm3
	vaddpd	%ymm11, %ymm9, %ymm12
	vmulpd	%ymm3, %ymm3, %ymm14
	vpermilpd	$2, %ymm0, %ymm9
	vaddpd	%ymm10, %ymm13, %ymm10
	vblendpd	$8, %ymm9, %ymm4, %ymm11
	vmulpd	%ymm11, %ymm11, %ymm6
	vaddpd	%ymm10, %ymm12, %ymm15
	vshufpd	$2, %ymm8, %ymm5, %ymm12
	vinsertf128	$1, %xmm0, %ymm12, %ymm13
	vblendpd	$7, %ymm13, %ymm0, %ymm10
	vmulpd	%ymm10, %ymm10, %ymm1
	vaddpd	%ymm6, %ymm14, %ymm0
	vaddpd	%ymm15, %ymm1, %ymm15
	vaddpd	%ymm15, %ymm0, %ymm10
	cmpq	%rdi, %r9
	jne	.L277
.L397:
	vextractf128	$0x1, %ymm10, %xmm5
	movl	%r11d, %r12d
	vaddpd	%xmm10, %xmm5, %xmm2
	andl	$-4, %r12d
	leal	(%r12,%r12,2), %eax
	vunpckhpd	%xmm2, %xmm2, %xmm7
	vaddpd	%xmm2, %xmm7, %xmm4
	cmpl	%r11d, %r12d
	je	.L404
.L276:
	leal	1(%rax), %ecx
	leal	2(%rax), %edi
	movslq	%eax, %r14
	movslq	%ecx, %rdx
	movslq	%edi, %r8
	vmovsd	0(%r13,%r14,8), %xmm12
	leal	3(%rax), %r9d
	vmovsd	0(%r13,%rdx,8), %xmm10
	vmovsd	0(%r13,%r8,8), %xmm14
	vmulsd	%xmm12, %xmm12, %xmm13
	vmulsd	%xmm10, %xmm10, %xmm6
	vmulsd	%xmm14, %xmm14, %xmm0
	vaddsd	%xmm4, %xmm13, %xmm4
	vaddsd	%xmm0, %xmm6, %xmm1
	vaddsd	%xmm4, %xmm1, %xmm7
	cmpl	%r9d, %r15d
	jle	.L279
	leal	4(%rax), %r14d
	leal	5(%rax), %edx
	movslq	%r9d, %r12
	movslq	%r14d, %rcx
	movslq	%edx, %rdi
	vmovsd	0(%r13,%r12,8), %xmm15
	leal	6(%rax), %r8d
	vmovsd	0(%r13,%rcx,8), %xmm3
	vmovsd	0(%r13,%rdi,8), %xmm2
	vmulsd	%xmm15, %xmm15, %xmm5
	vmulsd	%xmm3, %xmm3, %xmm8
	vmulsd	%xmm2, %xmm2, %xmm9
	vaddsd	%xmm7, %xmm5, %xmm7
	vaddsd	%xmm9, %xmm8, %xmm11
	vaddsd	%xmm7, %xmm11, %xmm7
	cmpl	%r8d, %r15d
	jle	.L279
	leal	7(%rax), %r12d
	addl	$8, %eax
	movslq	%r8d, %r9
	movslq	%r12d, %r14
	cltq
	vmovsd	0(%r13,%r9,8), %xmm12
	vmovsd	0(%r13,%r14,8), %xmm10
	vmovsd	0(%r13,%rax,8), %xmm14
	vmulsd	%xmm12, %xmm12, %xmm13
	vmulsd	%xmm14, %xmm14, %xmm6
	vmulsd	%xmm10, %xmm10, %xmm0
	vaddsd	%xmm7, %xmm13, %xmm4
	vaddsd	%xmm0, %xmm6, %xmm1
	vaddsd	%xmm4, %xmm1, %xmm7
.L279:
	leal	-3(%rsi,%rsi,2), %esi
	vxorpd	%xmm15, %xmm15, %xmm15
	vcvtsi2sdl	%esi, %xmm15, %xmm5
	vmulsd	Tinit(%rip), %xmm5, %xmm3
	vdivsd	%xmm7, %xmm3, %xmm11
	vsqrtsd	%xmm11, %xmm11, %xmm11
	cmpl	$8, %r10d
	jbe	.L290
.L286:
	movl	%r11d, %eax
	vmovddup	%xmm11, %xmm8
	movabsq	$192153584101141163, %rdx
	shrl	$2, %eax
	vinsertf128	$1, %xmm8, %ymm8, %ymm8
	leaq	(%rax,%rax,2), %r10
	salq	$5, %r10
	leaq	(%r10,%rbx), %rcx
	subq	$96, %r10
	shrq	$5, %r10
	imulq	%rdx, %r10
	addq	$1, %r10
	andl	$7, %r10d
	je	.L282
	cmpq	$1, %r10
	je	.L362
	cmpq	$2, %r10
	je	.L363
	cmpq	$3, %r10
	je	.L364
	cmpq	$4, %r10
	je	.L365
	cmpq	$5, %r10
	je	.L366
	cmpq	$6, %r10
	jne	.L405
.L367:
	vmulpd	32(%rbx), %ymm8, %ymm13
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm10
	vmulpd	-96(%rbx), %ymm8, %ymm4
	vmovapd	%ymm13, -64(%rbx)
	vmovapd	%ymm10, -32(%rbx)
	vmovapd	%ymm4, -96(%rbx)
.L366:
	vmulpd	32(%rbx), %ymm8, %ymm14
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm6
	vmulpd	-96(%rbx), %ymm8, %ymm0
	vmovapd	%ymm14, -64(%rbx)
	vmovapd	%ymm6, -32(%rbx)
	vmovapd	%ymm0, -96(%rbx)
.L365:
	vmulpd	32(%rbx), %ymm8, %ymm1
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm7
	vmulpd	-96(%rbx), %ymm8, %ymm15
	vmovapd	%ymm1, -64(%rbx)
	vmovapd	%ymm7, -32(%rbx)
	vmovapd	%ymm15, -96(%rbx)
.L364:
	vmulpd	32(%rbx), %ymm8, %ymm5
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm3
	vmulpd	-96(%rbx), %ymm8, %ymm9
	vmovapd	%ymm5, -64(%rbx)
	vmovapd	%ymm3, -32(%rbx)
	vmovapd	%ymm9, -96(%rbx)
.L363:
	vmulpd	32(%rbx), %ymm8, %ymm12
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm2
	vmulpd	-96(%rbx), %ymm8, %ymm13
	vmovapd	%ymm12, -64(%rbx)
	vmovapd	%ymm2, -32(%rbx)
	vmovapd	%ymm13, -96(%rbx)
.L362:
	vmulpd	32(%rbx), %ymm8, %ymm10
	addq	$96, %rbx
	vmulpd	-32(%rbx), %ymm8, %ymm14
	vmulpd	-96(%rbx), %ymm8, %ymm4
	vmovapd	%ymm10, -64(%rbx)
	vmovapd	%ymm14, -32(%rbx)
	vmovapd	%ymm4, -96(%rbx)
	cmpq	%rbx, %rcx
	je	.L300
.L282:
	vmulpd	32(%rbx), %ymm8, %ymm6
	addq	$768, %rbx
	vmulpd	-704(%rbx), %ymm8, %ymm0
	vmulpd	-768(%rbx), %ymm8, %ymm1
	vmulpd	-608(%rbx), %ymm8, %ymm15
	vmulpd	-672(%rbx), %ymm8, %ymm5
	vmulpd	-544(%rbx), %ymm8, %ymm3
	vmovapd	%ymm6, -736(%rbx)
	vmulpd	-512(%rbx), %ymm8, %ymm9
	vmovapd	%ymm0, -704(%rbx)
	vmulpd	-576(%rbx), %ymm8, %ymm12
	vmovapd	%ymm1, -768(%rbx)
	vmulpd	-448(%rbx), %ymm8, %ymm13
	vmovapd	%ymm15, -608(%rbx)
	vmulpd	-640(%rbx), %ymm8, %ymm7
	vmovapd	%ymm5, -672(%rbx)
	vmulpd	-416(%rbx), %ymm8, %ymm2
	vmovapd	%ymm3, -544(%rbx)
	vmulpd	-480(%rbx), %ymm8, %ymm10
	vmovapd	%ymm9, -512(%rbx)
	vmulpd	-352(%rbx), %ymm8, %ymm14
	vmovapd	%ymm12, -576(%rbx)
	vmulpd	-320(%rbx), %ymm8, %ymm6
	vmovapd	%ymm13, -448(%rbx)
	vmulpd	-384(%rbx), %ymm8, %ymm4
	vmovapd	%ymm7, -640(%rbx)
	vmulpd	-256(%rbx), %ymm8, %ymm0
	vmovapd	%ymm2, -416(%rbx)
	vmulpd	-224(%rbx), %ymm8, %ymm1
	vmovapd	%ymm10, -480(%rbx)
	vmovapd	%ymm14, -352(%rbx)
	vmovapd	%ymm6, -320(%rbx)
	vmovapd	%ymm4, -384(%rbx)
	vmulpd	-288(%rbx), %ymm8, %ymm7
	vmulpd	-160(%rbx), %ymm8, %ymm15
	vmovapd	%ymm0, -256(%rbx)
	vmulpd	-128(%rbx), %ymm8, %ymm5
	vmovapd	%ymm1, -224(%rbx)
	vmulpd	-192(%rbx), %ymm8, %ymm3
	vmulpd	-64(%rbx), %ymm8, %ymm9
	vmulpd	-32(%rbx), %ymm8, %ymm12
	vmovapd	%ymm7, -288(%rbx)
	vmulpd	-96(%rbx), %ymm8, %ymm13
	vmovapd	%ymm15, -160(%rbx)
	vmovapd	%ymm5, -128(%rbx)
	vmovapd	%ymm3, -192(%rbx)
	vmovapd	%ymm9, -64(%rbx)
	vmovapd	%ymm12, -32(%rbx)
	vmovapd	%ymm13, -96(%rbx)
	cmpq	%rbx, %rcx
	jne	.L282
.L300:
	movl	%r11d, %ebx
	andl	$-4, %ebx
	leal	(%rbx,%rbx,2), %eax
	cmpl	%r11d, %ebx
	je	.L400
.L284:
	movslq	%eax, %r11
	leal	1(%rax), %edi
	leal	2(%rax), %r9d
	vmulsd	0(%r13,%r11,8), %xmm11, %xmm8
	movslq	%edi, %r8
	leal	3(%rax), %r14d
	movslq	%r9d, %r12
	vmovsd	%xmm8, 0(%r13,%r11,8)
	vmulsd	0(%r13,%r8,8), %xmm11, %xmm2
	vmovsd	%xmm2, 0(%r13,%r8,8)
	vmulsd	0(%r13,%r12,8), %xmm11, %xmm10
	vmovsd	%xmm10, 0(%r13,%r12,8)
	cmpl	%r14d, %r15d
	jle	.L400
	movslq	%r14d, %rsi
	leal	4(%rax), %r10d
	leal	5(%rax), %edx
	vmulsd	0(%r13,%rsi,8), %xmm11, %xmm14
	movslq	%r10d, %rcx
	leal	6(%rax), %r11d
	movslq	%edx, %rbx
	vmovsd	%xmm14, 0(%r13,%rsi,8)
	vmulsd	0(%r13,%rcx,8), %xmm11, %xmm6
	vmovsd	%xmm6, 0(%r13,%rcx,8)
	vmulsd	0(%r13,%rbx,8), %xmm11, %xmm4
	vmovsd	%xmm4, 0(%r13,%rbx,8)
	cmpl	%r11d, %r15d
	jle	.L400
	movslq	%r11d, %r15
	leal	7(%rax), %edi
	addl	$8, %eax
	vmulsd	0(%r13,%r15,8), %xmm11, %xmm0
	movslq	%edi, %r8
	cltq
	vmovsd	%xmm0, 0(%r13,%r15,8)
	vmulsd	0(%r13,%r8,8), %xmm11, %xmm1
	vmovsd	%xmm1, 0(%r13,%r8,8)
	vmulsd	0(%r13,%rax,8), %xmm11, %xmm11
	vmovsd	%xmm11, 0(%r13,%rax,8)
	vzeroupper
.L401:
	addq	$32, %rsp
	popq	%rbx
	popq	%rax
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%rax), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L400:
	.cfi_restore_state
	vzeroupper
	jmp	.L401
	.p2align 4,,10
	.p2align 3
.L404:
	leal	-3(%rsi,%rsi,2), %r10d
	vxorpd	%xmm3, %xmm3, %xmm3
	vcvtsi2sdl	%r10d, %xmm3, %xmm8
	vmulsd	Tinit(%rip), %xmm8, %xmm9
	vdivsd	%xmm4, %xmm9, %xmm11
	vsqrtsd	%xmm11, %xmm11, %xmm11
	jmp	.L286
	.p2align 4,,10
	.p2align 3
.L403:
	movl	N(%rip), %esi
	vxorpd	%xmm14, %xmm14, %xmm14
	vmovsd	.LC2(%rip), %xmm7
	vcvtsi2sdl	%esi, %xmm14, %xmm6
	vdivsd	%xmm6, %xmm7, %xmm8
	vmulsd	%xmm2, %xmm8, %xmm2
	vmulsd	%xmm15, %xmm8, %xmm0
	vmulsd	%xmm1, %xmm8, %xmm1
	jmp	.L287
	.p2align 4,,10
	.p2align 3
.L402:
	vmovapd	(%rcx), %ymm7
	vmovapd	32+v(%rip), %ymm0
	leaq	96+v(%rip), %rcx
	vmovapd	64+v(%rip), %ymm5
	jmp	.L361
	.p2align 4,,10
	.p2align 3
.L405:
	vmulpd	(%rbx), %ymm8, %ymm12
	addq	$96, %rbx
	vmulpd	32+v(%rip), %ymm8, %ymm9
	vmulpd	64+v(%rip), %ymm8, %ymm2
	vmovapd	%ymm12, -96(%rbx)
	vmovapd	%ymm9, 32+v(%rip)
	vmovapd	%ymm2, 64+v(%rip)
	jmp	.L367
.L288:
	vxorpd	%xmm15, %xmm15, %xmm15
	xorl	%eax, %eax
	vmovapd	%xmm15, %xmm2
	vmovapd	%xmm15, %xmm1
	jmp	.L268
.L289:
	vxorpd	%xmm4, %xmm4, %xmm4
	xorl	%eax, %eax
	jmp	.L276
.L291:
	xorl	%eax, %eax
	jmp	.L285
.L290:
	xorl	%eax, %eax
	jmp	.L284
	.cfi_endproc
.LFE5591:
	.size	_Z20initializeVelocitiesv, .-_Z20initializeVelocitiesv
	.p2align 4
	.globl	_Z10initializev
	.type	_Z10initializev, @function
_Z10initializev:
.LFB5584:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
1:	call	*mcount@GOTPCREL(%rip)
	movl	N(%rip), %eax
	vxorps	%xmm3, %xmm3, %xmm3
	vcvtsi2sdl	%eax, %xmm3, %xmm0
	leal	(%rax,%rax,2), %ebx
	call	cbrt@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	vmovsd	L(%rip), %xmm1
	vroundsd	$10, %xmm0, %xmm0, %xmm0
	vcvttsd2sil	%xmm0, %r11d
	vcvtsi2sdl	%r11d, %xmm3, %xmm0
	vdivsd	%xmm0, %xmm1, %xmm1
	testl	%r11d, %r11d
	jle	.L407
	leal	-1(%r11), %edx
	leal	(%r11,%r11,2), %r12d
	xorl	%r13d, %r13d
	xorl	%r8d, %r8d
	vmovsd	.LC1(%rip), %xmm2
	leaq	3(%rdx,%rdx,2), %r10
	leaq	r(%rip), %rsi
	.p2align 4,,10
	.p2align 3
.L411:
	vcvtsi2sdl	%r13d, %xmm3, %xmm5
	xorl	%edi, %edi
	vaddsd	%xmm2, %xmm5, %xmm4
	vmulsd	%xmm1, %xmm4, %xmm5
	.p2align 4,,10
	.p2align 3
.L410:
	vcvtsi2sdl	%edi, %xmm3, %xmm6
	leaq	-3(%r10), %rcx
	movslq	%r8d, %rax
	xorl	%edx, %edx
	movabsq	$-6148914691236517205, %r14
	leaq	(%r10,%rax), %r9
	imulq	%r14, %rcx
	vaddsd	%xmm2, %xmm6, %xmm7
	addq	$1, %rcx
	vmulsd	%xmm1, %xmm7, %xmm8
	vunpcklpd	%xmm8, %xmm5, %xmm9
	andl	$7, %ecx
	je	.L409
	cmpq	$1, %rcx
	je	.L450
	cmpq	$2, %rcx
	je	.L451
	cmpq	$3, %rcx
	je	.L452
	cmpq	$4, %rcx
	je	.L453
	cmpq	$5, %rcx
	je	.L454
	cmpq	$6, %rcx
	je	.L455
	cmpl	%eax, %ebx
	jle	.L415
	vmulsd	%xmm1, %xmm2, %xmm10
	leal	2(%rax), %edx
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%edx, %rcx
	vmovsd	%xmm10, (%rsi,%rcx,8)
.L415:
	movl	$1, %edx
	addq	$3, %rax
.L455:
	cmpl	%eax, %ebx
	jle	.L418
	vcvtsi2sdl	%edx, %xmm3, %xmm11
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm11, %xmm12
	vmulsd	%xmm1, %xmm12, %xmm13
	vmovsd	%xmm13, (%rsi,%rcx,8)
.L418:
	addl	$1, %edx
	addq	$3, %rax
.L454:
	cmpl	%eax, %ebx
	jle	.L421
	vcvtsi2sdl	%edx, %xmm3, %xmm14
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm14, %xmm15
	vmulsd	%xmm1, %xmm15, %xmm0
	vmovsd	%xmm0, (%rsi,%rcx,8)
.L421:
	addl	$1, %edx
	addq	$3, %rax
.L453:
	cmpl	%eax, %ebx
	jle	.L424
	vcvtsi2sdl	%edx, %xmm3, %xmm4
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm4, %xmm6
	vmulsd	%xmm1, %xmm6, %xmm7
	vmovsd	%xmm7, (%rsi,%rcx,8)
.L424:
	addl	$1, %edx
	addq	$3, %rax
.L452:
	cmpl	%eax, %ebx
	jle	.L427
	vcvtsi2sdl	%edx, %xmm3, %xmm8
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm8, %xmm10
	vmulsd	%xmm1, %xmm10, %xmm11
	vmovsd	%xmm11, (%rsi,%rcx,8)
.L427:
	addl	$1, %edx
	addq	$3, %rax
.L451:
	cmpl	%eax, %ebx
	jle	.L430
	vcvtsi2sdl	%edx, %xmm3, %xmm12
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm12, %xmm13
	vmulsd	%xmm1, %xmm13, %xmm14
	vmovsd	%xmm14, (%rsi,%rcx,8)
.L430:
	addl	$1, %edx
	addq	$3, %rax
.L450:
	cmpl	%eax, %ebx
	jle	.L433
	vcvtsi2sdl	%edx, %xmm3, %xmm15
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm15, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm4
	vmovsd	%xmm4, (%rsi,%rcx,8)
.L433:
	addq	$3, %rax
	addl	$1, %edx
	cmpq	%rax, %r9
	je	.L465
.L409:
	cmpl	%eax, %ebx
	jle	.L408
	vcvtsi2sdl	%edx, %xmm3, %xmm6
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm6, %xmm7
	vmulsd	%xmm1, %xmm7, %xmm8
	vmovsd	%xmm8, (%rsi,%rcx,8)
.L408:
	addq	$3, %rax
	addl	$1, %edx
	cmpl	%eax, %ebx
	jle	.L436
	vcvtsi2sdl	%edx, %xmm3, %xmm10
	leal	2(%rax), %r14d
	vmovups	%xmm9, (%rsi,%rax,8)
	movslq	%r14d, %rcx
	vaddsd	%xmm2, %xmm10, %xmm11
	vmulsd	%xmm1, %xmm11, %xmm12
	vmovsd	%xmm12, (%rsi,%rcx,8)
.L436:
	leaq	3(%rax), %r14
	leal	1(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L438
	vcvtsi2sdl	%ecx, %xmm3, %xmm13
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm13, %xmm14
	vmulsd	%xmm1, %xmm14, %xmm15
	vmovsd	%xmm15, (%rsi,%r14,8)
.L438:
	leaq	6(%rax), %r14
	leal	2(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L440
	vcvtsi2sdl	%ecx, %xmm3, %xmm0
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm0, %xmm4
	vmulsd	%xmm1, %xmm4, %xmm6
	vmovsd	%xmm6, (%rsi,%r14,8)
.L440:
	leaq	9(%rax), %r14
	leal	3(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L442
	vcvtsi2sdl	%ecx, %xmm3, %xmm7
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm7, %xmm8
	vmulsd	%xmm1, %xmm8, %xmm10
	vmovsd	%xmm10, (%rsi,%r14,8)
.L442:
	leaq	12(%rax), %r14
	leal	4(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L444
	vcvtsi2sdl	%ecx, %xmm3, %xmm11
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm11, %xmm12
	vmulsd	%xmm1, %xmm12, %xmm13
	vmovsd	%xmm13, (%rsi,%r14,8)
.L444:
	leaq	15(%rax), %r14
	leal	5(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L446
	vcvtsi2sdl	%ecx, %xmm3, %xmm14
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm14, %xmm15
	vmulsd	%xmm1, %xmm15, %xmm0
	vmovsd	%xmm0, (%rsi,%r14,8)
.L446:
	leaq	18(%rax), %r14
	leal	6(%rdx), %ecx
	cmpl	%r14d, %ebx
	jle	.L448
	vcvtsi2sdl	%ecx, %xmm3, %xmm4
	vmovups	%xmm9, (%rsi,%r14,8)
	addl	$2, %r14d
	movslq	%r14d, %r14
	vaddsd	%xmm2, %xmm4, %xmm6
	vmulsd	%xmm1, %xmm6, %xmm7
	vmovsd	%xmm7, (%rsi,%r14,8)
.L448:
	addq	$21, %rax
	addl	$7, %edx
	cmpq	%rax, %r9
	jne	.L409
.L465:
	leal	1(%rdi), %eax
	addl	%r12d, %r8d
	cmpl	%eax, %r11d
	je	.L466
	movl	%eax, %edi
	jmp	.L410
	.p2align 4,,10
	.p2align 3
.L466:
	leal	1(%r13), %r9d
	cmpl	%r13d, %edi
	je	.L407
	movl	%r9d, %r13d
	jmp	.L411
.L407:
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_def_cfa 7, 8
	jmp	_Z20initializeVelocitiesv
	.cfi_endproc
.LFE5584:
	.size	_Z10initializev, .-_Z10initializev
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC32:
	.string	"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	.align 8
.LC33:
	.string	"                  WELCOME TO WILLY P CHEM MD!"
	.align 8
.LC34:
	.string	"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	.align 8
.LC35:
	.string	"\n  ENTER A TITLE FOR YOUR CALCULATION!"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC36:
	.string	"%s"
.LC37:
	.string	"_traj.xyz"
.LC38:
	.string	"_output.txt"
.LC39:
	.string	"_average.txt"
	.section	.rodata.str1.8
	.align 8
.LC40:
	.string	"                  TITLE ENTERED AS '%s'\n"
	.align 8
.LC41:
	.string	"  WHICH NOBLE GAS WOULD YOU LIKE TO SIMULATE? (DEFAULT IS ARGON)"
	.align 8
.LC42:
	.string	"\n  FOR HELIUM,  TYPE 'He' THEN PRESS 'return' TO CONTINUE"
	.align 8
.LC43:
	.string	"  FOR NEON,    TYPE 'Ne' THEN PRESS 'return' TO CONTINUE"
	.align 8
.LC44:
	.string	"  FOR ARGON,   TYPE 'Ar' THEN PRESS 'return' TO CONTINUE"
	.align 8
.LC45:
	.string	"  FOR KRYPTON, TYPE 'Kr' THEN PRESS 'return' TO CONTINUE"
	.align 8
.LC46:
	.string	"  FOR XENON,   TYPE 'Xe' THEN PRESS 'return' TO CONTINUE"
	.section	.rodata.str1.1
.LC47:
	.string	"He"
.LC48:
	.string	"Ne"
.LC49:
	.string	"Ar"
.LC50:
	.string	"Kr"
.LC51:
	.string	"Xe"
	.section	.rodata.str1.8
	.align 8
.LC52:
	.string	"\n                     YOU ARE SIMULATING %s GAS! \n"
	.align 8
.LC53:
	.string	"\n  YOU WILL NOW ENTER A FEW SIMULATION PARAMETERS"
	.align 8
.LC54:
	.string	"\n\n  ENTER THE INTIAL TEMPERATURE OF YOUR GAS IN KELVIN"
	.section	.rodata.str1.1
.LC55:
	.string	"%lf"
	.section	.rodata.str1.8
	.align 8
.LC56:
	.string	"\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!"
	.align 8
.LC57:
	.string	"\n\n  ENTER THE NUMBER DENSITY IN moles/m^3"
	.align 8
.LC58:
	.string	"  FOR REFERENCE, NUMBER DENSITY OF AN IDEAL GAS AT STP IS ABOUT 40 moles/m^3"
	.align 8
.LC59:
	.string	"  NUMBER DENSITY OF LIQUID ARGON AT 1 ATM AND 87 K IS ABOUT 35000 moles/m^3"
	.align 8
.LC61:
	.string	"\n\n\n  YOUR DENSITY IS VERY HIGH!\n"
	.align 8
.LC62:
	.string	"  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n"
	.align 8
.LC63:
	.string	"  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE"
	.align 8
.LC64:
	.string	"  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n"
	.section	.rodata.str1.1
.LC65:
	.string	"w"
.LC68:
	.string	"%i\n"
	.section	.rodata.str1.8
	.align 8
.LC69:
	.string	"  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n"
	.align 8
.LC70:
	.string	"  PERCENTAGE OF CALCULATION COMPLETE:\n  ["
	.section	.rodata.str1.1
.LC71:
	.string	" 10 |"
.LC72:
	.string	" 20 |"
.LC73:
	.string	" 30 |"
.LC74:
	.string	" 40 |"
.LC75:
	.string	" 50 |"
.LC76:
	.string	" 60 |"
.LC77:
	.string	" 70 |"
.LC78:
	.string	" 80 |"
.LC79:
	.string	" 90 |"
.LC80:
	.string	" 100 ]"
	.section	.rodata.str1.8
	.align 8
.LC82:
	.string	"  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n"
	.align 8
.LC83:
	.string	"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n"
	.align 8
.LC84:
	.string	" --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n"
	.align 8
.LC85:
	.string	"  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n"
	.align 8
.LC86:
	.string	"\n  TO ANIMATE YOUR SIMULATION, OPEN THE FILE \n  '%s' WITH VMD AFTER THE SIMULATION COMPLETES\n"
	.align 8
.LC87:
	.string	"\n  TO ANALYZE INSTANTANEOUS DATA ABOUT YOUR MOLECULE, OPEN THE FILE \n  '%s' WITH YOUR FAVORITE TEXT EDITOR OR IMPORT THE DATA INTO EXCEL\n"
	.align 8
.LC88:
	.string	"\n  THE FOLLOWING THERMODYNAMIC AVERAGES WILL BE COMPUTED AND WRITTEN TO THE FILE  \n  '%s':\n"
	.align 8
.LC89:
	.string	"\n  AVERAGE TEMPERATURE (K):                 %15.5f\n"
	.align 8
.LC90:
	.string	"\n  AVERAGE PRESSURE  (Pa):                  %15.5f\n"
	.align 8
.LC91:
	.string	"\n  PV/nT (J * mol^-1 K^-1):                 %15.5f\n"
	.align 8
.LC94:
	.string	"\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.5f\n"
	.align 8
.LC95:
	.string	"\n  THE COMPRESSIBILITY (unitless):          %15.5f \n"
	.align 8
.LC96:
	.string	"\n  TOTAL VOLUME (m^3):                      %10.5e \n"
	.align 8
.LC97:
	.string	"\n  NUMBER OF PARTICLES (unitless):          %i \n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5583:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$4096, %rsp
	orq	$0, (%rsp)
	addq	$-128, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
1:	call	*mcount@GOTPCREL(%rip)
	leaq	.LC32(%rip), %rdi
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	leaq	-4080(%rbp), %r12
	leaq	-3072(%rbp), %rbx
	call	puts@PLT
	leaq	.LC33(%rip), %rdi
	leaq	-2064(%rbp), %r13
	call	puts@PLT
	leaq	.LC34(%rip), %rdi
	leaq	-1056(%rbp), %r14
	call	puts@PLT
	leaq	.LC35(%rip), %rdi
	call	puts@PLT
	movq	%r12, %rsi
	leaq	.LC36(%rip), %rdi
	xorl	%eax, %eax
	call	__isoc99_scanf@PLT
	movl	$1000, %edx
	movq	%r12, %rsi
	movq	%rbx, %rdi
	movq	%rbx, -4224(%rbp)
	call	__stpcpy_chk@PLT
	movq	%rbx, %rcx
	movl	$10, %edx
	leaq	.LC37(%rip), %rsi
	subq	%rax, %rcx
	movq	%rax, %rdi
	addq	$1000, %rcx
	call	__memcpy_chk@PLT
	movl	$1000, %edx
	movq	%r12, %rsi
	movq	%r13, %rdi
	movq	%r13, -4208(%rbp)
	call	__stpcpy_chk@PLT
	movq	%r13, %rcx
	movl	$12, %edx
	leaq	.LC38(%rip), %rsi
	subq	%rax, %rcx
	movq	%rax, %rdi
	addq	$1000, %rcx
	call	__memcpy_chk@PLT
	movl	$1000, %edx
	movq	%r12, %rsi
	movq	%r14, %rdi
	movq	%r14, -4216(%rbp)
	call	__stpcpy_chk@PLT
	movq	%r14, %rcx
	movl	$13, %edx
	leaq	.LC39(%rip), %rsi
	subq	%rax, %rcx
	movq	%rax, %rdi
	addq	$1000, %rcx
	call	__memcpy_chk@PLT
	leaq	.LC32(%rip), %rdi
	call	puts@PLT
	movq	%r12, %rdx
	movl	$1, %edi
	xorl	%eax, %eax
	leaq	.LC40(%rip), %rsi
	call	__printf_chk@PLT
	leaq	.LC34(%rip), %rdi
	call	puts@PLT
	leaq	.LC32(%rip), %rdi
	call	puts@PLT
	leaq	.LC41(%rip), %rdi
	call	puts@PLT
	leaq	.LC42(%rip), %rdi
	call	puts@PLT
	leaq	.LC43(%rip), %rdi
	call	puts@PLT
	leaq	.LC44(%rip), %rdi
	call	puts@PLT
	leaq	.LC45(%rip), %rdi
	call	puts@PLT
	leaq	.LC46(%rip), %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rdi
	call	puts@PLT
	leaq	atype(%rip), %rsi
	leaq	.LC36(%rip), %rdi
	xorl	%eax, %eax
	call	__isoc99_scanf@PLT
	cmpw	$25928, atype(%rip)
	je	.L560
.L468:
	leaq	.LC48(%rip), %rsi
	leaq	atype(%rip), %rdi
	call	strcmp@PLT
	testl	%eax, %eax
	jne	.L561
	vmovsd	.LC16(%rip), %xmm2
	vmovsd	.LC17(%rip), %xmm0
	vmovsd	.LC18(%rip), %xmm1
	vmovsd	.LC19(%rip), %xmm3
	vmovsd	%xmm2, -4104(%rbp)
	vmovsd	%xmm0, -4160(%rbp)
	vmovsd	%xmm1, -4152(%rbp)
	vmovsd	%xmm3, -4232(%rbp)
.L470:
	leaq	.LC32(%rip), %rdi
	call	puts@PLT
	leaq	atype(%rip), %rdx
	leaq	.LC52(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	call	__printf_chk@PLT
	leaq	.LC32(%rip), %rdi
	call	puts@PLT
	leaq	.LC32(%rip), %rdi
	call	puts@PLT
	leaq	.LC53(%rip), %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rdi
	call	puts@PLT
	leaq	.LC54(%rip), %rdi
	call	puts@PLT
	xorl	%eax, %eax
	leaq	Tinit(%rip), %rsi
	leaq	.LC55(%rip), %rdi
	call	__isoc99_scanf@PLT
	vmovsd	Tinit(%rip), %xmm4
	vxorpd	%xmm5, %xmm5, %xmm5
	vcomisd	%xmm4, %xmm5
	ja	.L562
	leaq	.LC57(%rip), %rdi
	vdivsd	-4152(%rbp), %xmm4, %xmm6
	vmovsd	%xmm6, Tinit(%rip)
	call	puts@PLT
	leaq	.LC58(%rip), %rdi
	call	puts@PLT
	leaq	.LC59(%rip), %rdi
	call	puts@PLT
	xorl	%eax, %eax
	leaq	-4088(%rbp), %rsi
	leaq	.LC55(%rip), %rdi
	call	__isoc99_scanf@PLT
	vmovsd	-4088(%rbp), %xmm7
	vmulsd	NA(%rip), %xmm7, %xmm8
	movl	$2160, N(%rip)
	vmulsd	-4232(%rbp), %xmm8, %xmm9
	vmovsd	.LC60(%rip), %xmm10
	vdivsd	%xmm9, %xmm10, %xmm11
	vcomisd	%xmm11, %xmm10
	vmovsd	%xmm11, -4240(%rbp)
	jbe	.L549
	leaq	.LC61(%rip), %rdi
	call	puts@PLT
	movl	N(%rip), %edx
	movl	$1, %edi
	vmovsd	-4240(%rbp), %xmm0
	leaq	.LC62(%rip), %rsi
	movl	$1, %eax
	call	__printf_chk@PLT
	leaq	.LC63(%rip), %rdi
	call	puts@PLT
	leaq	.LC64(%rip), %rdi
	call	puts@PLT
	xorl	%edi, %edi
	call	exit@PLT
.L561:
	leaq	.LC49(%rip), %rsi
	leaq	atype(%rip), %rdi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L500
	leaq	.LC50(%rip), %rsi
	leaq	atype(%rip), %rdi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L501
	leaq	.LC51(%rip), %rsi
	leaq	atype(%rip), %rdi
	call	strcmp@PLT
	testl	%eax, %eax
	jne	.L563
	vmovsd	.LC28(%rip), %xmm4
	vmovsd	.LC29(%rip), %xmm5
	vmovsd	.LC30(%rip), %xmm6
	vmovsd	.LC31(%rip), %xmm7
	vmovsd	%xmm4, -4104(%rbp)
	vmovsd	%xmm5, -4160(%rbp)
	vmovsd	%xmm6, -4152(%rbp)
	vmovsd	%xmm7, -4232(%rbp)
	jmp	.L470
.L562:
	leaq	.LC56(%rip), %rdi
	call	puts@PLT
	xorl	%edi, %edi
	call	exit@PLT
.L560:
	cmpb	$0, 2+atype(%rip)
	jne	.L468
	vmovsd	.LC24(%rip), %xmm2
	vmovsd	.LC25(%rip), %xmm0
	vmovsd	.LC26(%rip), %xmm1
	vmovsd	.LC27(%rip), %xmm3
	vmovsd	%xmm2, -4104(%rbp)
	vmovsd	%xmm0, -4160(%rbp)
	vmovsd	%xmm1, -4152(%rbp)
	vmovsd	%xmm3, -4232(%rbp)
	jmp	.L470
.L549:
	vmovsd	-4240(%rbp), %xmm0
	call	cbrt@PLT
	movq	-4224(%rbp), %rdi
	leaq	.LC65(%rip), %rsi
	vmovsd	%xmm0, L(%rip)
	call	fopen@PLT
	movq	-4208(%rbp), %rdi
	leaq	.LC65(%rip), %rsi
	movq	%rax, %r14
	call	fopen@PLT
	movq	-4216(%rbp), %rdi
	leaq	.LC65(%rip), %rsi
	movq	%rax, %r13
	call	fopen@PLT
	leaq	.LC47(%rip), %rsi
	leaq	atype(%rip), %rdi
	movq	%rax, -4272(%rbp)
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L564
	movl	$200, -4248(%rbp)
	vmovsd	.LC67(%rip), %xmm12
	vdivsd	-4104(%rbp), %xmm12, %xmm13
	vmovsd	%xmm13, -4184(%rbp)
.L476:
	call	_Z10initializev
	call	_Z20computeAccelerationsv
	movl	N(%rip), %ecx
	movq	%r14, %rdi
	xorl	%eax, %eax
	leaq	.LC68(%rip), %rdx
	movl	$1, %esi
	call	__fprintf_chk@PLT
	movl	-4248(%rbp), %eax
	movl	$10, %ecx
	movl	$1, %esi
	leaq	.LC69(%rip), %rdi
	cltd
	idivl	%ecx
	movq	%r13, %rcx
	movl	$127, %edx
	movl	%eax, -4172(%rbp)
	call	fwrite@PLT
	leaq	.LC70(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	-4172(%rbp), %edi
	vmovsd	-4184(%rbp), %xmm2
	movq	$0x000000000, -4112(%rbp)
	vmulsd	-4104(%rbp), %xmm2, %xmm0
	movl	-4248(%rbp), %eax
	movq	$0x000000000, -4104(%rbp)
	imull	$6, %edi, %r15d
	leal	(%rdi,%rdi,8), %ebx
	imull	$7, %edi, %r10d
	leal	(%rdi,%rdi), %esi
	addl	$1, %eax
	movl	%ebx, -4260(%rbp)
	imull	$10, %edi, %ecx
	leal	(%rdi,%rdi,2), %r8d
	movl	%esi, -4188(%rbp)
	xorl	%ebx, %ebx
	leal	0(,%rdi,4), %r9d
	leal	(%rdi,%rdi,4), %r11d
	movl	%r15d, -4244(%rbp)
	leal	0(,%rdi,8), %r12d
	movl	%r8d, -4192(%rbp)
	leaq	v(%rip), %r15
	movl	%r9d, -4196(%rbp)
	movl	%r11d, -4200(%rbp)
	movl	%r10d, -4252(%rbp)
	movl	%r12d, -4256(%rbp)
	movl	%ecx, -4264(%rbp)
	movl	%eax, -4176(%rbp)
	vmovsd	%xmm0, -4168(%rbp)
	cmpl	%ebx, -4172(%rbp)
	je	.L565
	.p2align 4,,10
	.p2align 3
.L477:
	cmpl	%ebx, -4188(%rbp)
	je	.L566
	cmpl	%ebx, -4192(%rbp)
	je	.L567
	cmpl	%ebx, -4196(%rbp)
	je	.L568
	cmpl	%ebx, -4200(%rbp)
	je	.L569
	cmpl	%ebx, -4244(%rbp)
	je	.L570
	cmpl	%ebx, -4252(%rbp)
	je	.L571
	cmpl	%ebx, -4256(%rbp)
	je	.L572
	cmpl	%ebx, -4260(%rbp)
	je	.L573
	cmpl	%ebx, -4264(%rbp)
	je	.L574
	.p2align 4,,10
	.p2align 3
.L478:
	movq	stdout(%rip), %rdi
	leal	1(%rbx), %r12d
	call	fflush@PLT
	vmovsd	-4184(%rbp), %xmm0
	movq	%r14, %rsi
	movl	%r12d, %edi
	call	_Z14VelocityVerletdiP8_IO_FILE
	movl	N(%rip), %edi
	vmulsd	-4160(%rbp), %xmm0, %xmm12
	leal	(%rdi,%rdi,2), %esi
	testl	%esi, %esi
	jle	.L487
	leal	-1(%rsi), %r8d
	cmpl	$2, %r8d
	jbe	.L503
	movl	%esi, %edx
	leaq	v(%rip), %r11
	vxorpd	%xmm2, %xmm2, %xmm2
	shrl	$2, %edx
	salq	$5, %rdx
	leaq	(%rdx,%r15), %r9
	subq	$32, %rdx
	shrq	$5, %rdx
	addq	$1, %rdx
	andl	$7, %edx
	je	.L489
	cmpq	$1, %rdx
	je	.L533
	cmpq	$2, %rdx
	je	.L534
	cmpq	$3, %rdx
	je	.L535
	cmpq	$4, %rdx
	je	.L536
	cmpq	$5, %rdx
	je	.L537
	cmpq	$6, %rdx
	je	.L538
	vmovapd	(%r15), %ymm1
	leaq	32(%r15), %r11
	vmulpd	%ymm1, %ymm1, %ymm2
.L538:
	vmovapd	(%r11), %ymm3
	addq	$32, %r11
	vmulpd	%ymm3, %ymm3, %ymm4
	vaddpd	%ymm4, %ymm2, %ymm2
.L537:
	vmovapd	(%r11), %ymm5
	addq	$32, %r11
	vmulpd	%ymm5, %ymm5, %ymm6
	vaddpd	%ymm6, %ymm2, %ymm2
.L536:
	vmovapd	(%r11), %ymm7
	addq	$32, %r11
	vmulpd	%ymm7, %ymm7, %ymm8
	vaddpd	%ymm8, %ymm2, %ymm2
.L535:
	vmovapd	(%r11), %ymm9
	addq	$32, %r11
	vmulpd	%ymm9, %ymm9, %ymm10
	vaddpd	%ymm10, %ymm2, %ymm2
.L534:
	vmovapd	(%r11), %ymm11
	addq	$32, %r11
	vmulpd	%ymm11, %ymm11, %ymm13
	vaddpd	%ymm13, %ymm2, %ymm2
.L533:
	vmovapd	(%r11), %ymm14
	addq	$32, %r11
	vmulpd	%ymm14, %ymm14, %ymm15
	vaddpd	%ymm15, %ymm2, %ymm2
	cmpq	%r9, %r11
	je	.L553
.L489:
	vmovapd	(%r11), %ymm0
	vmovapd	32(%r11), %ymm4
	addq	$256, %r11
	vmovapd	-192(%r11), %ymm6
	vmovapd	-128(%r11), %ymm14
	vmovapd	-160(%r11), %ymm10
	vmulpd	%ymm0, %ymm0, %ymm1
	vmovapd	-96(%r11), %ymm0
	vmulpd	%ymm4, %ymm4, %ymm5
	vmovapd	-64(%r11), %ymm4
	vmulpd	%ymm6, %ymm6, %ymm8
	vmovapd	-32(%r11), %ymm6
	vmulpd	%ymm10, %ymm10, %ymm11
	vmulpd	%ymm14, %ymm14, %ymm15
	vaddpd	%ymm1, %ymm2, %ymm3
	vmulpd	%ymm0, %ymm0, %ymm1
	vaddpd	%ymm5, %ymm3, %ymm7
	vmulpd	%ymm4, %ymm4, %ymm5
	vaddpd	%ymm8, %ymm7, %ymm9
	vmulpd	%ymm6, %ymm6, %ymm8
	vaddpd	%ymm11, %ymm9, %ymm13
	vaddpd	%ymm15, %ymm13, %ymm2
	vaddpd	%ymm1, %ymm2, %ymm3
	vaddpd	%ymm5, %ymm3, %ymm7
	vaddpd	%ymm8, %ymm7, %ymm2
	cmpq	%r9, %r11
	jne	.L489
.L553:
	vextractf128	$0x1, %ymm2, %xmm9
	movl	%esi, %eax
	vaddpd	%xmm2, %xmm9, %xmm10
	andl	$-4, %eax
	vunpckhpd	%xmm10, %xmm10, %xmm11
	vaddpd	%xmm10, %xmm11, %xmm0
	testb	$3, %sil
	je	.L490
.L488:
	movslq	%eax, %r10
	leal	1(%rax), %ecx
	vmovsd	(%r15,%r10,8), %xmm13
	vmulsd	%xmm13, %xmm13, %xmm14
	vaddsd	%xmm14, %xmm0, %xmm0
	cmpl	%ecx, %esi
	jle	.L490
	movslq	%ecx, %rdx
	addl	$2, %eax
	vmovsd	(%r15,%rdx,8), %xmm15
	vmulsd	%xmm15, %xmm15, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	cmpl	%eax, %esi
	jle	.L490
	cltq
	vmovsd	(%r15,%rax,8), %xmm1
	vmulsd	%xmm1, %xmm1, %xmm3
	vaddsd	%xmm3, %xmm0, %xmm0
.L490:
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	%r8d, %r9d
	vcvtsi2sdl	%edi, %xmm4, %xmm5
	movl	$2863311531, %edi
	imulq	%rdi, %r9
	vdivsd	%xmm5, %xmm0, %xmm7
	shrq	$33, %r9
	vmovsd	m(%rip), %xmm5
	vmulsd	.LC1(%rip), %xmm5, %xmm6
	addl	$1, %r9d
	vmovsd	%xmm7, -4144(%rbp)
	cmpl	$8, %r8d
	jbe	.L505
	movl	%r9d, %r8d
	leaq	v(%rip), %rcx
	vmovddup	%xmm6, %xmm13
	shrl	$2, %r8d
	vinsertf128	$1, %xmm13, %ymm13, %ymm13
	vxorpd	%xmm0, %xmm0, %xmm0
	leaq	(%r8,%r8,2), %r11
	salq	$5, %r11
	addq	%r15, %r11
	movq	%r11, %r10
	subq	%rcx, %r10
	andl	$32, %r10d
	je	.L493
	vmovapd	v(%rip), %ymm10
	vmovapd	64+v(%rip), %ymm11
	leaq	96+v(%rip), %rcx
	vperm2f128	$48, 32+v(%rip), %ymm10, %ymm14
	vperm2f128	$2, 32+v(%rip), %ymm10, %ymm3
	vperm2f128	$33, %ymm10, %ymm10, %ymm8
	vpermilpd	$2, %ymm11, %ymm0
	vshufpd	$2, %ymm8, %ymm14, %ymm15
	vperm2f128	$33, %ymm11, %ymm15, %ymm9
	vshufpd	$5, %ymm3, %ymm14, %ymm1
	vblendpd	$8, %ymm0, %ymm1, %ymm4
	vshufpd	$2, %ymm3, %ymm8, %ymm7
	vmulpd	%ymm4, %ymm4, %ymm14
	vblendpd	$8, %ymm9, %ymm15, %ymm2
	vinsertf128	$1, %xmm11, %ymm7, %ymm10
	vmulpd	%ymm2, %ymm2, %ymm8
	vblendpd	$7, %ymm10, %ymm11, %ymm11
	vmulpd	%ymm11, %ymm11, %ymm9
	vaddpd	%ymm8, %ymm14, %ymm15
	vaddpd	%ymm9, %ymm15, %ymm2
	vmulpd	%ymm13, %ymm2, %ymm0
	cmpq	%rcx, %r11
	je	.L539
	.p2align 4,,10
	.p2align 3
.L493:
	vmovapd	(%rcx), %ymm4
	vmovapd	64(%rcx), %ymm7
	addq	$192, %rcx
	vperm2f128	$48, -160(%rcx), %ymm4, %ymm1
	vperm2f128	$2, -160(%rcx), %ymm4, %ymm8
	vperm2f128	$33, %ymm4, %ymm4, %ymm11
	vpermilpd	$2, %ymm7, %ymm9
	vshufpd	$2, %ymm11, %ymm1, %ymm3
	vperm2f128	$33, %ymm7, %ymm3, %ymm10
	vshufpd	$5, %ymm8, %ymm1, %ymm15
	vblendpd	$8, %ymm9, %ymm15, %ymm2
	vblendpd	$8, %ymm10, %ymm3, %ymm14
	vshufpd	$2, %ymm8, %ymm11, %ymm4
	vinsertf128	$1, %xmm7, %ymm4, %ymm1
	vmovapd	-96(%rcx), %ymm4
	vmulpd	%ymm2, %ymm2, %ymm11
	vblendpd	$7, %ymm1, %ymm7, %ymm7
	vmulpd	%ymm14, %ymm14, %ymm3
	vperm2f128	$33, %ymm4, %ymm4, %ymm9
	vmulpd	%ymm7, %ymm7, %ymm14
	vmovapd	-32(%rcx), %ymm7
	vaddpd	%ymm3, %ymm11, %ymm10
	vmovapd	-64(%rcx), %ymm3
	vperm2f128	$48, %ymm3, %ymm4, %ymm1
	vaddpd	%ymm14, %ymm10, %ymm8
	vpermilpd	$2, %ymm7, %ymm10
	vmulpd	%ymm13, %ymm8, %ymm15
	vperm2f128	$2, %ymm3, %ymm4, %ymm8
	vshufpd	$2, %ymm8, %ymm9, %ymm3
	vaddpd	%ymm15, %ymm0, %ymm2
	vshufpd	$2, %ymm9, %ymm1, %ymm0
	vperm2f128	$33, %ymm7, %ymm0, %ymm11
	vshufpd	$5, %ymm8, %ymm1, %ymm15
	vblendpd	$8, %ymm11, %ymm0, %ymm14
	vblendpd	$8, %ymm10, %ymm15, %ymm4
	vinsertf128	$1, %xmm7, %ymm3, %ymm1
	vmulpd	%ymm14, %ymm14, %ymm0
	vblendpd	$7, %ymm1, %ymm7, %ymm7
	vmulpd	%ymm4, %ymm4, %ymm9
	vmulpd	%ymm7, %ymm7, %ymm14
	vaddpd	%ymm0, %ymm9, %ymm11
	vaddpd	%ymm14, %ymm11, %ymm8
	vmulpd	%ymm13, %ymm8, %ymm15
	vaddpd	%ymm15, %ymm2, %ymm0
	cmpq	%rcx, %r11
	jne	.L493
.L539:
	vextractf128	$0x1, %ymm0, %xmm13
	movl	%r9d, %edx
	vaddpd	%xmm0, %xmm13, %xmm2
	andl	$-4, %edx
	leal	(%rdx,%rdx,2), %eax
	vunpckhpd	%xmm2, %xmm2, %xmm10
	vaddpd	%xmm2, %xmm10, %xmm3
	cmpl	%r9d, %edx
	je	.L559
.L495:
	leal	1(%rax), %edi
	leal	2(%rax), %r11d
	movslq	%eax, %r9
	movslq	%edi, %r8
	movslq	%r11d, %r10
	vmovsd	(%r15,%r9,8), %xmm1
	leal	3(%rax), %ecx
	vmovsd	(%r15,%r8,8), %xmm7
	vmovsd	(%r15,%r10,8), %xmm4
	vmulsd	%xmm1, %xmm1, %xmm14
	vmulsd	%xmm7, %xmm7, %xmm9
	vmulsd	%xmm4, %xmm4, %xmm0
	vaddsd	%xmm0, %xmm9, %xmm11
	vaddsd	%xmm14, %xmm11, %xmm8
	vmulsd	%xmm6, %xmm8, %xmm15
	vaddsd	%xmm15, %xmm3, %xmm3
	cmpl	%ecx, %esi
	jle	.L559
	leal	4(%rax), %r9d
	leal	5(%rax), %r8d
	movslq	%ecx, %rdx
	movslq	%r9d, %rdi
	movslq	%r8d, %r11
	vmovsd	(%r15,%rdx,8), %xmm13
	leal	6(%rax), %r10d
	vmovsd	(%r15,%rdi,8), %xmm2
	vmovsd	(%r15,%r11,8), %xmm10
	vmulsd	%xmm13, %xmm13, %xmm9
	vmulsd	%xmm2, %xmm2, %xmm1
	vmulsd	%xmm10, %xmm10, %xmm7
	vaddsd	%xmm7, %xmm1, %xmm4
	vaddsd	%xmm9, %xmm4, %xmm0
	vmulsd	%xmm6, %xmm0, %xmm11
	vaddsd	%xmm11, %xmm3, %xmm3
	cmpl	%r10d, %esi
	jle	.L559
	leal	7(%rax), %ecx
	addl	$8, %eax
	movslq	%r10d, %rsi
	movslq	%ecx, %rdx
	cltq
	vmovsd	(%r15,%rsi,8), %xmm14
	vmovsd	(%r15,%rdx,8), %xmm8
	vmovsd	(%r15,%rax,8), %xmm15
	vmulsd	%xmm14, %xmm14, %xmm1
	vmulsd	%xmm15, %xmm15, %xmm13
	vmulsd	%xmm8, %xmm8, %xmm2
	vaddsd	%xmm2, %xmm13, %xmm10
	vaddsd	%xmm1, %xmm10, %xmm7
	vmulsd	%xmm6, %xmm7, %xmm6
	vaddsd	%xmm6, %xmm3, %xmm3
	vzeroupper
.L496:
	vmovsd	%xmm3, -4136(%rbp)
	vmovsd	%xmm5, -4128(%rbp)
	vmovsd	%xmm12, -4120(%rbp)
	call	_Z9Potentialv
	vmovsd	-4128(%rbp), %xmm12
	movq	%r13, %rdi
	vxorpd	%xmm15, %xmm15, %xmm15
	vmulsd	-4144(%rbp), %xmm12, %xmm9
	vcvtsi2sdl	%ebx, %xmm15, %xmm13
	vmovapd	%xmm0, %xmm4
	vmovsd	-4136(%rbp), %xmm3
	vmulsd	.LC81(%rip), %xmm9, %xmm0
	movl	$1, %esi
	movl	$6, %eax
	vmovsd	-4120(%rbp), %xmm2
	vdivsd	kB(%rip), %xmm0, %xmm11
	vmulsd	-4168(%rbp), %xmm13, %xmm0
	vaddsd	%xmm3, %xmm4, %xmm5
	vaddsd	-4112(%rbp), %xmm2, %xmm8
	vmulsd	-4152(%rbp), %xmm11, %xmm1
	leaq	.LC82(%rip), %rdx
	vaddsd	-4104(%rbp), %xmm1, %xmm14
	vmovsd	%xmm8, -4112(%rbp)
	vmovsd	%xmm14, -4104(%rbp)
	call	__fprintf_chk@PLT
	cmpl	-4176(%rbp), %r12d
	je	.L575
	movl	%r12d, %ebx
	cmpl	%ebx, -4172(%rbp)
	jne	.L477
.L565:
	leaq	.LC71(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L564:
	vmovsd	.LC66(%rip), %xmm14
	movl	$50000, -4248(%rbp)
	vdivsd	-4104(%rbp), %xmm14, %xmm15
	vmovsd	%xmm15, -4184(%rbp)
	jmp	.L476
.L563:
	movw	$29249, atype(%rip)
	movb	$0, 2+atype(%rip)
.L500:
	vmovsd	.LC12(%rip), %xmm12
	vmovsd	.LC13(%rip), %xmm13
	vmovsd	.LC14(%rip), %xmm14
	vmovsd	.LC15(%rip), %xmm15
	vmovsd	%xmm12, -4104(%rbp)
	vmovsd	%xmm13, -4160(%rbp)
	vmovsd	%xmm14, -4152(%rbp)
	vmovsd	%xmm15, -4232(%rbp)
	jmp	.L470
	.p2align 4,,10
	.p2align 3
.L559:
	vzeroupper
	jmp	.L496
.L566:
	leaq	.LC72(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L567:
	leaq	.LC73(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L575:
	vxorpd	%xmm10, %xmm10, %xmm10
	vmovsd	-4104(%rbp), %xmm7
	vmovsd	-4112(%rbp), %xmm4
	movl	$115, %edx
	vcvtsi2sdl	-4248(%rbp), %xmm10, %xmm1
	vmovsd	-4232(%rbp), %xmm5
	vmulsd	-4240(%rbp), %xmm5, %xmm12
	movl	$1, %esi
	vcvtsi2sdl	N(%rip), %xmm10, %xmm9
	vmulsd	NA(%rip), %xmm12, %xmm8
	movq	-4272(%rbp), %r15
	leaq	.LC83(%rip), %rdi
	vmulsd	kBSI(%rip), %xmm9, %xmm3
	movq	%r15, %rcx
	vdivsd	%xmm1, %xmm7, %xmm6
	vmovsd	%xmm12, -4128(%rbp)
	vdivsd	%xmm1, %xmm4, %xmm2
	vmulsd	%xmm6, %xmm3, %xmm11
	vmovsd	%xmm6, -4112(%rbp)
	vmulsd	%xmm6, %xmm9, %xmm13
	vmulsd	%xmm2, %xmm12, %xmm0
	vmovsd	%xmm2, -4120(%rbp)
	vmulsd	%xmm2, %xmm8, %xmm15
	vdivsd	%xmm11, %xmm0, %xmm14
	vdivsd	%xmm13, %xmm15, %xmm10
	vmovsd	%xmm14, -4144(%rbp)
	vmovsd	%xmm10, -4104(%rbp)
	call	fwrite@PLT
	movq	%r15, %rcx
	movl	$117, %edx
	movl	$1, %esi
	leaq	.LC84(%rip), %rdi
	call	fwrite@PLT
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%r15, %rdi
	vmovsd	-4128(%rbp), %xmm5
	vcvtsi2sdl	%r12d, %xmm1, %xmm4
	vmulsd	-4168(%rbp), %xmm4, %xmm0
	movl	N(%rip), %ecx
	leaq	.LC85(%rip), %rdx
	vmovsd	-4144(%rbp), %xmm4
	vmovsd	-4104(%rbp), %xmm3
	movl	$1, %esi
	movl	$6, %eax
	vmovsd	-4112(%rbp), %xmm1
	vmovsd	-4120(%rbp), %xmm2
	vmovsd	%xmm5, -4136(%rbp)
	vmovsd	%xmm4, -4128(%rbp)
	call	__fprintf_chk@PLT
	movq	-4224(%rbp), %rdx
	leaq	.LC86(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	call	__printf_chk@PLT
	movq	-4208(%rbp), %rdx
	leaq	.LC87(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	call	__printf_chk@PLT
	movq	-4216(%rbp), %rdx
	leaq	.LC88(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4112(%rbp), %xmm0
	leaq	.LC89(%rip), %rsi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4120(%rbp), %xmm0
	leaq	.LC90(%rip), %rsi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4104(%rbp), %xmm0
	leaq	.LC91(%rip), %rsi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4104(%rbp), %xmm2
	vsubsd	.LC92(%rip), %xmm2, %xmm7
	vandpd	.LC8(%rip), %xmm7, %xmm6
	leaq	.LC94(%rip), %rsi
	vmulsd	.LC93(%rip), %xmm6, %xmm0
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4128(%rbp), %xmm0
	leaq	.LC95(%rip), %rsi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	-4136(%rbp), %xmm0
	leaq	.LC96(%rip), %rsi
	call	__printf_chk@PLT
	movl	N(%rip), %edx
	leaq	.LC97(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	call	__printf_chk@PLT
	movq	%r14, %rdi
	call	fclose@PLT
	movq	%r13, %rdi
	call	fclose@PLT
	movq	%r15, %rdi
	call	fclose@PLT
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L576
	addq	$4224, %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r9
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r9), %rsp
	.cfi_def_cfa 7, 8
	ret
.L568:
	.cfi_restore_state
	leaq	.LC74(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L487:
	movq	$0x000000000, -4144(%rbp)
	vmovsd	m(%rip), %xmm5
	vxorpd	%xmm3, %xmm3, %xmm3
	jmp	.L496
.L569:
	leaq	.LC75(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L505:
	vxorpd	%xmm3, %xmm3, %xmm3
	xorl	%eax, %eax
	jmp	.L495
.L503:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	jmp	.L488
.L570:
	leaq	.LC76(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L571:
	leaq	.LC77(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L572:
	leaq	.LC78(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L573:
	leaq	.LC79(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	jmp	.L478
.L574:
	leaq	.LC80(%rip), %rdi
	call	puts@PLT
	jmp	.L478
.L501:
	vmovsd	.LC20(%rip), %xmm8
	vmovsd	.LC21(%rip), %xmm9
	vmovsd	.LC22(%rip), %xmm10
	vmovsd	.LC23(%rip), %xmm11
	vmovsd	%xmm8, -4104(%rbp)
	vmovsd	%xmm9, -4160(%rbp)
	vmovsd	%xmm10, -4152(%rbp)
	vmovsd	%xmm11, -4232(%rbp)
	jmp	.L470
.L576:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5583:
	.size	main, .-main
	.text
	.p2align 4
	.globl	_Z9gaussdistv
	.type	_Z9gaussdistv, @function
_Z9gaussdistv:
.LFB5592:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
1:	call	*mcount@GOTPCREL(%rip)
	cmpb	$0, _ZZ9gaussdistvE9available(%rip)
	je	.L583
	vmovsd	_ZZ9gaussdistvE4gset(%rip), %xmm0
	movb	$0, _ZZ9gaussdistvE9available(%rip)
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L583:
	.cfi_restore_state
	call	rand@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sdl	%eax, %xmm4, %xmm2
	vmulsd	.LC10(%rip), %xmm2, %xmm0
	vsubsd	.LC2(%rip), %xmm0, %xmm1
	vmovsd	%xmm1, -8(%rbp)
	call	rand@PLT
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	-8(%rbp), %xmm8
	vmovsd	.LC2(%rip), %xmm12
	vcvtsi2sdl	%eax, %xmm3, %xmm5
	vmulsd	.LC10(%rip), %xmm5, %xmm6
	vsubsd	.LC2(%rip), %xmm6, %xmm7
	vmulsd	%xmm8, %xmm8, %xmm9
	vmulsd	%xmm7, %xmm7, %xmm10
	vaddsd	%xmm10, %xmm9, %xmm11
	vcomisd	%xmm11, %xmm12
	jbe	.L583
	vxorpd	%xmm13, %xmm13, %xmm13
	vcomisd	%xmm13, %xmm11
	je	.L583
	vmovapd	%xmm11, %xmm0
	vmovsd	%xmm7, -24(%rbp)
	vmovsd	%xmm8, -16(%rbp)
	vmovsd	%xmm11, -8(%rbp)
	call	log@PLT
	vmovsd	-8(%rbp), %xmm15
	vmovsd	-16(%rbp), %xmm2
	movb	$1, _ZZ9gaussdistvE9available(%rip)
	vmulsd	.LC11(%rip), %xmm0, %xmm14
	vmovsd	-24(%rbp), %xmm1
	vdivsd	%xmm15, %xmm14, %xmm4
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vmulsd	%xmm4, %xmm2, %xmm0
	vmovsd	%xmm0, _ZZ9gaussdistvE4gset(%rip)
	vmulsd	%xmm4, %xmm1, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5592:
	.size	_Z9gaussdistv, .-_Z9gaussdistv
	.local	_ZZ9gaussdistvE4gset
	.comm	_ZZ9gaussdistvE4gset,8,8
	.local	_ZZ9gaussdistvE9available
	.comm	_ZZ9gaussdistvE9available,1,1
	.globl	atype
	.bss
	.align 8
	.type	atype, @object
	.size	atype, 10
atype:
	.zero	10
	.globl	F
	.align 32
	.type	F, @object
	.size	F, 120024
F:
	.zero	120024
	.globl	a
	.align 32
	.type	a, @object
	.size	a, 120024
a:
	.zero	120024
	.globl	v
	.align 32
	.type	v, @object
	.size	v, 120024
v:
	.zero	120024
	.globl	r
	.align 32
	.type	r, @object
	.size	r, 120024
r:
	.zero	120024
	.globl	Tinit
	.align 8
	.type	Tinit, @object
	.size	Tinit, 8
Tinit:
	.zero	8
	.globl	L
	.align 8
	.type	L, @object
	.size	L, 8
L:
	.zero	8
	.globl	kBSI
	.data
	.align 8
	.type	kBSI, @object
	.size	kBSI, 8
kBSI:
	.long	1946377754
	.long	993046758
	.globl	NA
	.align 8
	.type	NA, @object
	.size	NA, 8
NA:
	.long	3539290983
	.long	1155522949
	.globl	kB
	.align 8
	.type	kB, @object
	.size	kB, 8
kB:
	.long	0
	.long	1072693248
	.globl	m
	.align 8
	.type	m, @object
	.size	m, 8
m:
	.long	0
	.long	1072693248
	.globl	epsilon
	.align 8
	.type	epsilon, @object
	.size	epsilon, 8
epsilon:
	.long	0
	.long	1072693248
	.globl	sigma
	.align 8
	.type	sigma, @object
	.size	sigma, 8
sigma:
	.long	0
	.long	1072693248
	.globl	N
	.bss
	.align 4
	.type	N, @object
	.size	N, 4
N:
	.zero	4
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	1071644672
	.align 8
.LC2:
	.long	0
	.long	1072693248
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC3:
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.long	0
	.long	-1074790400
	.section	.rodata.cst8
	.align 8
.LC4:
	.long	0
	.long	1074790400
	.align 8
.LC5:
	.long	0
	.long	1077411840
	.section	.rodata.cst32
	.align 32
.LC6:
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC7:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
	.align 16
.LC8:
	.long	4294967295
	.long	2147483647
	.long	0
	.long	0
	.section	.rodata.cst8
	.align 8
.LC9:
	.long	1431655765
	.long	1069897045
	.align 8
.LC10:
	.long	2097152
	.long	1041235968
	.align 8
.LC11:
	.long	0
	.long	-1073741824
	.align 8
.LC12:
	.long	2146467417
	.long	1031958573
	.align 8
.LC13:
	.long	143199393
	.long	1099474547
	.align 8
.LC14:
	.long	1030792151
	.long	1080148746
	.align 8
.LC15:
	.long	138146213
	.long	973606333
	.align 8
.LC16:
	.long	1452163321
	.long	1031971863
	.align 8
.LC17:
	.long	3832345398
	.long	1098511934
	.align 8
.LC18:
	.long	1486681760
	.long	1078216643
	.align 8
.LC19:
	.long	847494823
	.long	972690333
	.align 8
.LC20:
	.long	1015062865
	.long	1030509620
	.align 8
.LC21:
	.long	590927106
	.long	1099732054
	.align 8
.LC22:
	.long	4145216638
	.long	1080616400
	.align 8
.LC23:
	.long	1249961180
	.long	973935845
	.align 8
.LC24:
	.long	1017698372
	.long	1031727626
	.align 8
.LC25:
	.long	3582187013
	.long	1096751415
	.align 8
.LC26:
	.long	1899656693
	.long	1076214426
	.align 8
.LC27:
	.long	3983602444
	.long	972509966
	.align 8
.LC28:
	.long	465392776
	.long	1030732686
	.align 8
.LC29:
	.long	1995008865
	.long	1100009644
	.align 8
.LC30:
	.long	1370536935
	.long	1081181401
	.align 8
.LC31:
	.long	3663117873
	.long	974218174
	.align 8
.LC60:
	.long	0
	.long	1084284928
	.align 8
.LC66:
	.long	2665960982
	.long	1021445039
	.align 8
.LC67:
	.long	2258709403
	.long	1022788763
	.align 8
.LC81:
	.long	1431655765
	.long	1070945621
	.align 8
.LC92:
	.long	3757690939
	.long	1075880192
	.align 8
.LC93:
	.long	1487527648
	.long	1076366834
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
