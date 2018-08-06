
inline gmatc_e operator""_j(long double j) { return gmatc_e(0, (float)j); }
inline gmatc_e operator""_j(unsigned long long j) { return gmatc_e(0, (float)j); }

inline gmatc_e operator+(const gmatc_e & l, const gmatc_e & r) {
	return gmatc_e(l.real + r.real, l.imag + r.imag);
}
inline gmatc_e operator+(unsigned long long l, const gmatc_e & r) { return (gmatc_e)(float)l + r; }
inline gmatc_e operator+(const gmatc_e & l, unsigned long long r) { return l + (gmatc_e)(float)r; }

inline gmatc_e operator-(const gmatc_e & l, const gmatc_e & r)
{
	return gmatc_e(l.real - r.real, l.imag - r.imag);
}
inline gmatc_e operator-(unsigned long long l, const gmatc_e & r) { return (gmatc_e)(float)l - r; }
inline gmatc_e operator-(const gmatc_e & l, unsigned long long r) { return l - (gmatc_e)(float)r; }

inline gmatc_e operator*(const gmatc_e & l, const gmatc_e & r)
{
	return gmatc_e((l.real * r.real) - (l.imag * r.imag), (l.real * r.imag) + (l.imag * r.real));
}
inline gmatc_e operator*(unsigned long long l, const gmatc_e & r)
{
	return gmatc_e(l * r.real, l * r.imag);
}
inline gmatc_e operator*(const gmatc_e & l, unsigned long long r)
{
	return gmatc_e(l.real * r, l.imag * r);
}

inline gmatc_e operator/(const gmatc_e & l, const gmatc_e & r)
{
	float div = r.real * r.real + r.imag * r.imag;
	gmatc_e rtn((l.real * r.real) + (l.imag * r.imag), (l.imag * r.real) - (l.real * r.imag));
	rtn.real /= div;
	rtn.imag /= div;
	return rtn;
}
inline gmatc_e operator/(unsigned long long l, const gmatc_e & r) { return (gmatc_e)(float)l / r; }
inline gmatc_e operator/(const gmatc_e & l, unsigned long long r)
{
	return gmatc_e(l.real / (float)r, l.imag / (float)r);
}

template<typename T>
inline std::basic_ostream<T> & operator<<(std::basic_ostream<T> & o, const gmatc_e & i)
{
	o << i.real;
	if (i.imag >= 0)
		o << '+' << i.imag;
	else
		o << i.real;
	o << 'j';
	return o;
}


class gmatc
{
private:
	gmatc_e * _data;
	gmat_frame _frame;
public:
	inline gmatc() {}
	inline gmatc(size_t height, size_t width, int device = 0)
	{
		auto exc = gmatc_create(this, height, width, device);
		CHECK_EXC(exc);
	}
	inline gmatc(size_t height, size_t width, const void * src, int device = 0)
	{
		auto exc = gmatc_create(this, height, width, device);
		CHECK_EXC(exc);
		exc = gmatc_copy_from(this, src);
		CHECK_EXC(exc);
	}
	template<typename T>
	inline gmatc(size_t height, size_t width, T begin, T end, int device = 0)
	{
		auto exc = gmatc_create(this, height, width, device);
		CHECK_EXC(exc);

		gmatc_e * src = new gmatc_e[height * width];
		size_t p = 0;
		for (T it = begin; it != end; it++, p++)
			src[p] = (gmatc_e)(*it);

		for (; p < height * width; p++)
			src[p] = 0.0f;

		exc = gmatc_copy_from(this, src);
		delete[] src;
		CHECK_EXC(exc);
	}
	template<typename T>
	inline gmatc(const std::initializer_list<std::initializer_list<T>> & init, int device = 0)
	{
		size_t height = init.size();
		size_t width = 0;
		for (auto & i : init)
			if (width < i.size())
				width = i.size();

		auto exc = gmatc_create(this, height, width, device);
		CHECK_EXC(exc);

		gmatc_e * src = new gmatc_e[height * width];

		size_t hp = 0;
		size_t wp = 0;

		for (auto & i : init)
		{
			wp = 0;
			for (auto & j : i)
				src[hp * width + wp++] = (gmatc_e)j;
			for (; wp < width; wp++)
				src[hp * width + wp] = 0.0f;
			hp++;
		}

		exc = gmatc_copy_from(this, src);
		delete[] src;
		CHECK_EXC(exc);
	}
	inline gmatc(size_t size, int device = 0)
	{
		auto exc = gmatc_create_identity(this, size, device);
		CHECK_EXC(exc);
	}
	inline gmatc(const gmat_frame & frame)
	{
		auto exc = gmatc_create(this, frame.height, frame.width, frame.device);
		CHECK_EXC(exc);
	}
	inline gmatc(const gmatc & o)
	{
		auto exc = gmatc_create(this, o._frame.height, o._frame.width, o._frame.device);
		CHECK_EXC(exc);
		exc = gmatc_copy(this, &o);
		CHECK_EXC(exc);
	}
	inline gmatc(gmatc && o)
		: _data(o._data), _frame(o._frame)
	{
		o._data = nullptr;
		o._frame.height = 0;
		o._frame.width = 0;
		o._frame.device = 0;
	}

	inline gmatc & operator= (const gmatc & o)
	{
		gmatc_destroy(this);
		auto exc = gmatc_create(this, o._frame.height, o._frame.width, o._frame.device);
		CHECK_EXC(exc);
		exc = gmatc_copy(this, &o);
		CHECK_EXC(exc);
	}

	inline gmatc & operator= (gmatc && o)
	{
		gmatc_destroy(this);
		_data = o._data;
		_frame = o._frame;
		o._data = nullptr;
		o._frame.height = 0;
		o._frame.width = 0;
		o._frame.device = 0;
		return *this;
	}

	inline ~gmatc()
	{
		gmatc_destroy(this);
	}

public:
	inline size_t get_size() const noexcept
	{
		size_t rtn;
		gmatc_get_size(this, &rtn);
		return rtn;
	}

	inline size_t get_height() const noexcept
	{
		return _frame.height;
	}

	inline size_t get_width() const noexcept
	{
		return _frame.width;
	}

	inline auto & copy_to(void * dst) const
	{
		auto exc = gmatc_copy_to(this, dst);
		CHECK_EXC(exc);
		return *this;
	}

	inline gmatc & copy_from(const void * src)
	{
		auto exc = gmatc_copy_from(this, src);
		CHECK_EXC(exc);
		return *this;
	}

	inline gmatc copy() const
	{
		gmatc rtn(_frame.height, _frame.width, _frame.device);
		auto exc = gmatc_copy(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	inline gmatc copy(int device) const
	{
		gmatc rtn(_frame.height, _frame.width, device);
		auto exc = gmatc_copy(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	inline gmatc transpose() const
	{
		gmatc rtn(_frame.width, _frame.height, _frame.device);
		auto exc = gmatc_transpose(&rtn, this);
		CHECK_EXC(exc);

		return rtn;
	}

	inline gmatc conjugate() const
	{
		gmatc rtn(_frame);
		auto exc = gmatc_conjugate(&rtn, this);
		CHECK_EXC(exc);
		return rtn;
	}

	friend gmatc operator+ (const gmatc &, const gmatc &);
	friend gmatc operator+ (const gmatc &, const gmatc_e &);
	friend gmatc operator+ (const gmatc_e &, const gmatc &);
	friend gmatc operator- (const gmatc &, const gmatc &);
	friend gmatc operator- (const gmatc &, const gmatc_e &);
	friend gmatc operator- (const gmatc_e &, const gmatc &);
	friend gmatc operator* (const gmatc &, const gmatc &);
	friend gmatc operator* (const gmatc &, const gmatc_e &);
	friend gmatc operator* (const gmatc_e &, const gmatc &);
	friend gmatc operator/ (const gmatc &, const gmatc &);
	friend gmatc operator/ (const gmatc &, const gmatc_e &);
	friend gmatc operator/ (const gmatc_e &, const gmatc &);
	friend gmatc operator% (const gmatc &, const gmatc &);
	friend gmatc operator~ (const gmatc &);
	template<typename T>
	friend std::basic_ostream<T> & operator<< (std::basic_ostream<T> &, const gmatc &);
};

inline gmatc operator+ (const gmatc & mat0, const gmatc & mat1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_add(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator+ (const gmatc & mat0, const gmatc_e & val1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_add_gmatc_e(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator+ (const gmatc_e & val0, const gmatc & mat1)
{
	gmatc rtn(mat1._frame);
	auto exc = gmatc_add_gmatc_e(&rtn, &mat1, val0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator- (const gmatc & mat0, const gmatc & mat1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_sub(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator- (const gmatc & mat0, const gmatc_e & val1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_sub_gmatc_e(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator- (const gmatc_e & val0, const gmatc & mat1)
{
	gmatc rtn(mat1._frame);
	auto exc = gmatc_sub_gmatc_e_r(&rtn, val0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator* (const gmatc & mat0, const gmatc & mat1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_mul(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator* (const gmatc & mat0, const gmatc_e & val1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_mul_gmatc_e(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator* (const gmatc_e & val0, const gmatc & mat1)
{
	gmatc rtn(mat1._frame);
	auto exc = gmatc_mul_gmatc_e(&rtn, &mat1, val0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator/ (const gmatc & mat0, const gmatc & mat1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_div(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator/ (const gmatc & mat0, const gmatc_e & val1)
{
	gmatc rtn(mat0._frame);
	auto exc = gmatc_div_gmatc_e(&rtn, &mat0, val1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator/ (const gmatc_e & val0, const gmatc & mat1)
{
	gmatc rtn(mat1._frame);
	auto exc = gmatc_div_gmatc_e_r(&rtn, val0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator% (const gmatc & mat0, const gmatc & mat1)
{
	gmatc rtn(mat0._frame.height, mat1._frame.width, mat0._frame.device);
	auto exc = gmatc_product(&rtn, &mat0, &mat1);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc operator~ (const gmatc & mat0)
{
	gmatc rtn(mat0._frame.width, mat0._frame.height, mat0._frame.device);
	auto exc = gmatc_transpose(&rtn, &mat0);
	CHECK_EXC(exc);
	return rtn;
}

inline gmatc & operator+= (gmatc & mat0, const gmatc & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmatc & operator+= (gmatc & mat0, const gmatc_e & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmatc & operator-= (gmatc & mat0, const gmatc & mat1)
{
	return mat0 = mat0 - mat1;
}

inline gmatc & operator-= (gmatc & mat0, const gmatc_e & mat1)
{
	return mat0 = mat0 - mat1;
}

inline gmatc & operator*= (gmatc & mat0, const gmatc & mat1)
{
	return mat0 = mat0 * mat1;
}

inline gmatc & operator*= (gmatc & mat0, const gmatc_e & mat1)
{
	return mat0 = mat0 * mat1;
}

inline gmatc & operator/= (gmatc & mat0, const gmatc & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmatc & operator/= (gmatc & mat0, const gmatc_e & mat1)
{
	return mat0 = mat0 + mat1;
}

inline gmatc & operator%= (gmatc & mat0, const gmatc & mat1)
{
	return mat0 = mat0 % mat1;
}

inline gmatc operator- (const gmatc & mat0)
{
	return (const gmatc_e &)0 - mat0;
}

inline gmatc operator+ (const gmatc & mat0)
{
	return mat0;
}

template<typename T>
inline std::basic_ostream<T> & operator<<(std::basic_ostream<T> & o, const gmatc & g)
{
	auto height = g._frame.height;
	auto width = g._frame.width;

	gmatc_e * tmp = new gmatc_e[height * width];
	g.copy_to(tmp);

	for (int i = 0; i < height; i++)
	{
		o << "{ ";
		for (int j = 0; j < width; j++)
		{
			o << tmp[i * width + j] << ", ";
		}
		o << "},";
		if (i != height - 1)
			o << endl;
	}
	return o;
}